"""ROS 2 adapter for the come-here state machine.

All decisions live in ``come_here_fsm.ComeHereFsm`` (pure Python, tested with
fake time). This node parses messages, feeds the FSM with monotonic
timestamps, publishes the commands it returns, and records one JSON line per
trial as evidence.

Subscribes:
  /come_here/wake_phrase        (std_msgs/String)
  /come_here/wake_detail        (std_msgs/String) JSON from audio_node: confidence, latency
  /come_here/person_detection   (std_msgs/Float64MultiArray)
                                [bearing, distance, confidence, detected,
                                 bbox_h_frac, distance_source, frame_age_s]
  /come_here/audio_direction    (std_msgs/Float64MultiArray) [azimuth, confidence]
  /come_here/face_detection     (come_here_msgs/FaceDetection)
  /come_here/estop              (std_msgs/Bool)
  /come_here/rotate_result      (std_msgs/String) JSON from go2_bridge_node when a turn ends
  /come_here/reset              (std_msgs/Bool)   operator: stand up from DONE (estop_console)
  /come_here/skill_request      (std_msgs/String) JSON; only when enable_skill_api is true

Publishes:
  /come_here/cmd_velocity        (std_msgs/Float64MultiArray) [vx, yaw_rate]
  /come_here/cmd_rotate          (std_msgs/Float64)
  /come_here/cmd_say             (std_msgs/String)
  /come_here/cmd_sit, cmd_stand  (std_msgs/Bool)
  /come_here/face_detect_request (std_msgs/Bool)
  /come_here/state               (std_msgs/String)
  /come_here/status              (std_msgs/String) JSON for the operator view, every tick
  /come_here/target_gate         (std_msgs/Float64MultiArray) [center_rad, half_width_rad]
  /come_here/trial_summary       (std_msgs/String) JSON, one message per finished trial
  /come_here/skill_result        (std_msgs/String) JSON, transient_local; only when
                                 enable_skill_api is true

Parameters: every ``FsmConfig`` field (documented in come_here_fsm.py and the
YAML configs), plus tick_rate_hz, trial_log_enabled, trial_log_dir, git_commit, and
enable_skill_api (default false: no skill topics exist and behavior is the baseline).
"""

import dataclasses
import datetime
import json
import math
import random
import time

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

from come_here_behavior.come_here_fsm import (
    MOTION_STATES,
    ComeHereFsm,
    FsmConfig,
    PersonObservation,
    State,
)
from come_here_behavior.node_runner import run_node
from come_here_behavior.trial_log import TrialLogWriter, git_revision
from come_here_msgs.msg import FaceDetection

__all__ = ['BehaviorNode', 'State', 'main']

# A wake_detail message within this window of a wake belongs to that wake.
WAKE_DETAIL_WINDOW_S = 2.0


class BehaviorNode(Node):
    def __init__(self, **kwargs):
        super().__init__('behavior_node', **kwargs)

        self.declare_parameter('tick_rate_hz', 10.0)
        self.declare_parameter('trial_log_enabled', True)
        self.declare_parameter('trial_log_dir', '~/come_here_trials')
        self.declare_parameter('git_commit', '')
        # Not an FsmConfig field, so the config snapshot in every trial record is unchanged.
        self.declare_parameter('enable_skill_api', False)
        for f in dataclasses.fields(FsmConfig):
            self.declare_parameter(f.name, f.default)

        config = FsmConfig(**{
            f.name: self.get_parameter(f.name).value for f in dataclasses.fields(FsmConfig)
        })
        # Raises ValueError on an inconsistent config: the node refuses to start.
        self._fsm = ComeHereFsm(config)
        self._config_snapshot = dataclasses.asdict(config)
        # Monotonic: FSM timeouts must not jump when the Jetson clock is set by hand.
        self._now = time.monotonic

        git_override = str(self.get_parameter('git_commit').value)
        self._git = (
            {'commit': git_override, 'dirty': None} if git_override
            else git_revision(__file__)
        )
        self._log_writer = None
        if self.get_parameter('trial_log_enabled').value:
            log_dir = str(self.get_parameter('trial_log_dir').value)
            try:
                self._log_writer = TrialLogWriter(log_dir)
            except OSError as exc:
                self.get_logger().error(f'Trial log disabled, cannot open {log_dir}: {exc}')

        self._session_id = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        self._trial_index = 0
        self._trial_meta = None
        self._pending_detail = None
        self._pending_detail_s = None
        self._tick_count = 0

        # -- Publishers --
        self._velocity_pub = self.create_publisher(
            Float64MultiArray, '/come_here/cmd_velocity', 10
        )
        self._rotate_pub = self.create_publisher(Float64, '/come_here/cmd_rotate', 10)
        self._say_pub = self.create_publisher(String, '/come_here/cmd_say', 10)
        self._sit_pub = self.create_publisher(Bool, '/come_here/cmd_sit', 10)
        self._stand_pub = self.create_publisher(Bool, '/come_here/cmd_stand', 10)
        self._face_req_pub = self.create_publisher(Bool, '/come_here/face_detect_request', 10)
        self._state_pub = self.create_publisher(String, '/come_here/state', 10)
        self._trial_pub = self.create_publisher(String, '/come_here/trial_summary', 10)
        self._status_pub = self.create_publisher(String, '/come_here/status', 10)
        self._gate_pub = self.create_publisher(Float64MultiArray, '/come_here/target_gate', 10)
        self._skill_api = bool(self.get_parameter('enable_skill_api').value)
        self._skill_pub = None
        if self._skill_api:
            self._skill_pub = self.create_publisher(
                String, '/come_here/skill_result',
                QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL))

        # -- Subscribers --
        self.create_subscription(String, '/come_here/wake_phrase', self._wake_cb, 10)
        self.create_subscription(String, '/come_here/wake_detail', self._wake_detail_cb, 10)
        self.create_subscription(
            Float64MultiArray, '/come_here/person_detection', self._person_cb, 10
        )
        self.create_subscription(
            Float64MultiArray, '/come_here/audio_direction', self._direction_cb, 10
        )
        self.create_subscription(FaceDetection, '/come_here/face_detection', self._face_cb, 10)
        self.create_subscription(Bool, '/come_here/estop', self._estop_cb, 10)
        self.create_subscription(String, '/come_here/rotate_result', self._rotate_result_cb, 10)
        self.create_subscription(Bool, '/come_here/reset', self._reset_cb, 10)
        if self._skill_api:
            self.create_subscription(
                String, '/come_here/skill_request', self._skill_request_cb, 10)

        rate = float(self.get_parameter('tick_rate_hz').value)
        self._timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            'behavior_node started: '
            f'skip_turn_to_sound={config.skip_turn_to_sound} '
            f'arrival_mode={config.arrival_mode} speed={config.approach_speed} '
            f'align/realign={config.approach_align_threshold_rad}/'
            f'{config.approach_realign_threshold_rad} rad '
            f'bbox_stop={config.bbox_stop_fraction} '
            f'max_walk={config.max_walk_distance_m} m '
            f'git={self._git["commit"][:10]} '
            f'trial_log={self._log_writer.path if self._log_writer else "off"}'
            + (' skill_api=ON' if self._skill_api else '')
        )

    # -- inputs --

    def _wake_cb(self, msg: String) -> None:
        now = self._now()
        started = not self._fsm.trial_active
        cmds = self._fsm.on_wake(msg.data, now)
        if started and self._fsm.trial_active:
            self._trial_index += 1
            wake = {'source': 'topic'}
            if (self._pending_detail is not None
                    and now - self._pending_detail_s <= WAKE_DETAIL_WINDOW_S):
                wake = dict(self._pending_detail, source='audio')
            self._pending_detail = None
            self._trial_meta = {
                'run_id': f'{self._session_id}-{self._trial_index:03d}',
                'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
                'start_s': now,
                'wake': wake,
            }
        self._apply(cmds)

    def _skill_request_cb(self, msg: String) -> None:
        now = self._now()
        try:
            req = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            req = None                         # the FSM rejects it as malformed
        started = not self._fsm.trial_active
        cmds = self._fsm.on_skill_request(req, now)
        if started and self._fsm.trial_active:
            self._trial_index += 1
            self._pending_detail = None
            self._trial_meta = {
                'run_id': f'{self._session_id}-{self._trial_index:03d}',
                'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
                'start_s': now,
                'wake': {'source': 'skill_api'},
            }
        self._apply(cmds)

    def _wake_detail_cb(self, msg: String) -> None:
        try:
            detail = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('Ignoring malformed wake_detail')
            return
        if not isinstance(detail, dict):
            return
        now = self._now()
        meta = self._trial_meta
        # Topics are not ordered relative to each other: the detail may arrive
        # just after the wake it describes.
        if (meta is not None and meta['wake'].get('source') == 'topic'
                and now - meta['start_s'] <= WAKE_DETAIL_WINDOW_S):
            meta['wake'] = dict(detail, source='audio')
        else:
            self._pending_detail = detail
            self._pending_detail_s = now

    def _person_cb(self, msg: Float64MultiArray) -> None:
        obs = PersonObservation.from_array(list(msg.data))
        self._apply(self._fsm.on_person(obs, self._now()))

    def _direction_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 2:
            return
        self._apply(
            self._fsm.on_direction(float(msg.data[0]), float(msg.data[1]), self._now())
        )

    def _face_cb(self, msg: FaceDetection) -> None:
        self._apply(self._fsm.on_face_result(
            bool(msg.face_present), self._now(), float(msg.center_x_norm)))

    def _rotate_result_cb(self, msg: String) -> None:
        try:
            result = json.loads(msg.data)
            target = float(result['target_rad'])
            turned = result.get('turned_rad')
            turned = None if turned is None else float(turned)
            reason = str(result.get('reason', '?'))
        except (ValueError, TypeError, KeyError) as exc:
            self.get_logger().warn(f'Ignoring malformed rotate_result: {exc}')
            return
        if not math.isfinite(target):
            return
        self._apply(self._fsm.on_rotate_result(target, turned, reason, self._now()))

    def _reset_cb(self, msg: Bool) -> None:
        if msg.data:
            self._apply(self._fsm.on_reset(self._now()))

    def _estop_cb(self, msg: Bool) -> None:
        self._apply(self._fsm.on_estop(bool(msg.data), self._now()))

    def _tick(self) -> None:
        now = self._now()
        self._apply(self._fsm.tick(now))
        state = String()
        state.data = self._fsm.state.name
        self._state_pub.publish(state)
        status = String()
        status.data = json.dumps(self._fsm.viewer_status(now))
        self._status_pub.publish(status)
        self._tick_count += 1
        if self._fsm.state in MOTION_STATES and self._tick_count % 10 == 0:
            status = self._fsm.status()
            self.get_logger().info(
                'approach: ' + ' '.join(f'{k}={v}' for k, v in status.items())
            )

    # -- outputs --

    def _apply(self, cmds) -> None:
        for line in cmds.log:
            self.get_logger().info(line)
        if cmds.velocity is not None:
            msg = Float64MultiArray()
            msg.data = [float(cmds.velocity[0]), float(cmds.velocity[1])]
            self._velocity_pub.publish(msg)
        if cmds.rotate_rad is not None:
            msg = Float64()
            msg.data = float(cmds.rotate_rad)
            self._rotate_pub.publish(msg)
        if cmds.say:
            choices = [c.strip() for c in cmds.say.split('|') if c.strip()]
            if choices:
                msg = String()
                msg.data = random.choice(choices)
                self._say_pub.publish(msg)
        if cmds.gate is not None:
            msg = Float64MultiArray()
            msg.data = [float(cmds.gate[0]), float(cmds.gate[1])]
            self._gate_pub.publish(msg)
        for flag, pub in ((cmds.sit, self._sit_pub), (cmds.stand, self._stand_pub),
                          (cmds.face_request, self._face_req_pub)):
            if flag:
                msg = Bool()
                msg.data = True
                pub.publish(msg)
        run_id = (self._trial_meta or {}).get('run_id')
        if cmds.trial_summary is not None:
            self._record_trial(cmds.trial_summary)
        for result in cmds.skill_results:
            if self._skill_pub is None:
                break
            msg = String()
            msg.data = json.dumps(dict(result, run_id=None if result['status'] == 'rejected'
                                       else run_id), sort_keys=True)
            self._skill_pub.publish(msg)
            self.get_logger().info(f'SKILL RESULT {msg.data}')

    def _record_trial(self, summary: dict) -> None:
        meta = self._trial_meta or {}
        self._trial_meta = None
        record = {
            'run_id': meta.get('run_id'),
            'started_at': meta.get('started_at'),
            'ended_at': datetime.datetime.now().isoformat(timespec='seconds'),
            'git_commit': self._git['commit'],
            'git_dirty': self._git['dirty'],
            'wake': meta.get('wake', {'source': 'unknown'}),
            'config': self._config_snapshot,
        }
        record.update(summary)
        msg = String()
        msg.data = json.dumps(record, sort_keys=True)
        self._trial_pub.publish(msg)
        self.get_logger().info(
            f'TRIAL {record["run_id"]}: success={record["success"]} '
            f'stop_reason={record["stop_reason"]} '
            f'final_bbox_h_frac={record["final_bbox_h_frac"]} '
            f'align={record["align_phases"]} walk={record["walk_phases"]} '
            f'lost={record["lost_events"]}'
        )
        if self._log_writer is not None:
            try:
                self._log_writer.append(record)
            except OSError as exc:
                self.get_logger().error(f'Could not write trial log: {exc}')

    def _stop_motion(self) -> None:
        """Publish a zero cmd_velocity: the same path ALIGN and WALK drive (C1)."""
        msg = Float64MultiArray()
        msg.data = [0.0, 0.0]
        self._velocity_pub.publish(msg)

    def destroy_node(self):
        """Stop and close out any running trial; run_node() calls this while the
        ROS context is still valid, so the stop reaches the bridge."""
        try:
            self._apply(self._fsm.shutdown(self._now()))
        except Exception:  # noqa: BLE001 - shutdown must never raise
            try:
                self._stop_motion()
            except Exception:  # noqa: BLE001
                pass
        super().destroy_node()


def main(args=None):
    run_node(BehaviorNode, args=args)


if __name__ == '__main__':
    main()
