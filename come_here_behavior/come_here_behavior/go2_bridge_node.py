"""Bridge node: translate behavior commands into GO2 Sport + audiohub API calls.

Motion safety rules live in ``come_here_behavior.motion_gate`` (pure Python,
unit-tested without ROS): command validation and clamping, rejection of
combined forward + yaw, a command watchdog, the latched operator e-stop, and an
inhibit used while the motion mode is unverified. This node turns gate
decisions into Sport API requests. It never sends SelectMode.

Subscribes:
  /come_here/cmd_velocity         (std_msgs/Float64MultiArray) [vx, yaw_rate] gait command
  /come_here/estop                (std_msgs/Bool)   True engages the latched e-stop, False releases it
  /come_here/cmd_rotate           (std_msgs/Float64) turn in place by this angle (+ = left);
                                  closed loop on odometry yaw, see rotate_controller.py
  <odom_topic>                    (nav_msgs/Odometry) robot yaw for closed-loop turns
  /come_here/cmd_sit              (std_msgs/Bool)   True triggers Sit, optional
  /come_here/cmd_stand            (std_msgs/Bool)   True triggers BalanceStand, optional
  /come_here/cmd_say              (std_msgs/String) phrase to play through the audiohub
  /api/motion_switcher/response   (unitree_api/msg/Response) CheckMode replies

Publishes:
  /api/sport/request              (unitree_api/msg/Request) Sport API requests, or
                                  /come_here/dry_run/sport_request when dry_run is true
  /api/motion_switcher/request    (unitree_api/msg/Request) read-only CheckMode (api 1001)
  /api/audiohub/request           (unitree_api/msg/Request) audiohub WAV streaming
  /come_here/bridge_status        (std_msgs/String) JSON once per second

Parameters:
  dry_run                  (bool,  False) send Sport requests to the dry-run topic only
  require_motion_mode      (str,   '')    'mcf' or 'ai': refuse motion until CheckMode reports it
  motion_mode_timeout_s    (float, 5.0)   log a refusal if no CheckMode reply by then
  motion_mode_recheck_s    (float, 5.0)   re-check period once verified
  max_vx                   (float, 1.0)   clamp on |vx| in m/s
  max_yaw_rate             (float, 2.5)   clamp on |yaw_rate| in rad/s
  reject_vx_above          (float, 2.0)   larger |vx| is treated as corrupt and stops
  reject_yaw_rate_above    (float, 4.0)   larger |yaw_rate| is treated as corrupt and stops
  allow_combined_motion    (bool,  False) accept vx and yaw_rate in the same command
  cmd_velocity_timeout_s   (float, 0.5)   watchdog: StopMove if cmd_velocity goes quiet
  republish_rate_hz        (float, 20.0)  Move republish rate (mcf gait latching needs 20 Hz)
  enable_rotate_command    (bool,  True)  accept /come_here/cmd_rotate
  enable_posture_commands  (bool,  True)  accept /come_here/cmd_sit and /come_here/cmd_stand
  cmd_z                    (float, 2.0)   rotation worker yaw rate, capped at max_yaw_rate
  deg_per_sec              (float, 90.0)  measured rotation speed at cmd_z, for the timed
                                          fallback when odometry is missing
  rotate_closed_loop       (bool,  True)  turn until odometry yaw has moved by the target
  odom_topic               (str)          nav_msgs/Odometry source, /utlidar/robot_odom
  odom_max_age_s           (float, 0.5)   older odometry ends a closed-loop turn (stop)
  rotate_deadband_rad      (float, 0.12)  done inside this remaining angle
  rotate_timeout_s         (float, 6.0)   hard bound on any turn
  rotate_prefer_ccw_beyond_rad (float, 2.6) targets beyond this turn left, the long way
  sit_api_id / stand_api_id / move_api_id / stop_move_api_id  Sport API ids (never 1001, Damp)
  wav_dir / wav_chunk_size_bytes / wav_chunk_delay_s          audiohub playback
"""

import base64
import datetime
import json
import math
import os
import random
import threading
import time

from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from unitree_api.msg import Request, Response

from come_here_behavior.motion_gate import MOVE, NONE, STOP, GateLimits, MotionGate
from come_here_behavior.motion_mode import (
    MOTION_SWITCHER_CHECK_MODE_API_ID,
    MOTION_SWITCHER_REQUEST_TOPIC,
    MOTION_SWITCHER_RESPONSE_TOPIC,
    VALID_MOTION_MODES,
    check_sport_api_ids,
    mode_verdict,
    parse_mode_response,
)
from come_here_behavior.node_runner import run_node
from come_here_behavior.rotate_controller import RotateController

SPORT_TOPIC = '/api/sport/request'
DRY_RUN_SPORT_TOPIC = '/come_here/dry_run/sport_request'


def make_req(api_id, params=None):
    """Build a unitree_api Request for Sport or audiohub, matching the demo helper."""
    msg = Request()
    msg.header.identity.api_id = api_id
    msg.header.identity.id = (
        int(datetime.datetime.now().timestamp() * 1000 % 2147483648)
        + random.randint(0, 999)
    )
    msg.header.lease.id = 0
    msg.header.policy.priority = 0
    msg.header.policy.noreply = False
    if params is not None:
        msg.parameter = json.dumps(params) if isinstance(params, dict) else str(params)
    else:
        msg.parameter = ''
    msg.binary = []
    return msg


class Go2BridgeNode(Node):
    def __init__(self, **kwargs):
        super().__init__('go2_bridge_node', **kwargs)

        # -- Parameters --
        self.declare_parameter('dry_run', False)
        self.declare_parameter('require_motion_mode', '')
        self.declare_parameter('motion_mode_timeout_s', 5.0)
        self.declare_parameter('motion_mode_recheck_s', 5.0)
        self.declare_parameter('cmd_z', 2.0)
        self.declare_parameter('deg_per_sec', 90.0)
        self.declare_parameter('sit_api_id', 1005)
        self.declare_parameter('stand_api_id', 1002)
        self.declare_parameter('move_api_id', 1008)
        self.declare_parameter('stop_move_api_id', 1003)
        self.declare_parameter(
            'wav_dir', '/home/unitree/come-here/come_here_audio/scripts'
        )
        self.declare_parameter('wav_chunk_size_bytes', 16384)
        self.declare_parameter('wav_chunk_delay_s', 0.15)
        # Clamp limits sit above every calibrated setpoint (approach vx 0.6,
        # align yaw 0.6, rotate cmd_z 2.0); the reject thresholds catch values
        # that can only come from corruption.
        self.declare_parameter('max_vx', 1.0)
        self.declare_parameter('max_yaw_rate', 2.5)
        self.declare_parameter('reject_vx_above', 2.0)
        self.declare_parameter('reject_yaw_rate_above', 4.0)
        # mcf degrades on combined forward + yaw (measured 2026-04-17).
        self.declare_parameter('allow_combined_motion', False)
        self.declare_parameter('cmd_velocity_timeout_s', 0.5)
        # 10 Hz Move publishing did not keep the mcf gait latched; 20 Hz did
        # (hardware, 2026-04-24).
        self.declare_parameter('republish_rate_hz', 20.0)
        self.declare_parameter('enable_rotate_command', True)
        self.declare_parameter('rotate_closed_loop', True)
        self.declare_parameter('odom_topic', '/utlidar/robot_odom')
        self.declare_parameter('odom_max_age_s', 0.5)
        self.declare_parameter('rotate_deadband_rad', 0.12)
        self.declare_parameter('rotate_timeout_s', 6.0)
        self.declare_parameter('rotate_prefer_ccw_beyond_rad', 2.6)
        self.declare_parameter('enable_posture_commands', True)

        p = self.get_parameter
        self._dry_run = bool(p('dry_run').value)
        self._deg_per_sec = float(p('deg_per_sec').value)
        self._sit_api_id = int(p('sit_api_id').value)
        self._stand_api_id = int(p('stand_api_id').value)
        self._move_api_id = int(p('move_api_id').value)
        self._stop_move_api_id = int(p('stop_move_api_id').value)
        # 1001 on the Sport topic is Damp: refuse to start with it configured.
        check_sport_api_ids(
            sit_api_id=self._sit_api_id, stand_api_id=self._stand_api_id,
            move_api_id=self._move_api_id, stop_move_api_id=self._stop_move_api_id,
        )
        self._required_mode = str(p('require_motion_mode').value).strip()
        if self._required_mode and self._required_mode not in VALID_MOTION_MODES:
            raise ValueError(
                f'require_motion_mode must be one of {VALID_MOTION_MODES} or empty, '
                f'got {self._required_mode!r}'
            )
        self._mode_timeout_s = float(p('motion_mode_timeout_s').value)
        self._mode_recheck_s = float(p('motion_mode_recheck_s').value)
        self._wav_dir = str(p('wav_dir').value)
        self._wav_chunk_size = int(p('wav_chunk_size_bytes').value)
        self._wav_chunk_delay_s = float(p('wav_chunk_delay_s').value)
        self._enable_rotate = bool(p('enable_rotate_command').value)
        self._enable_posture = bool(p('enable_posture_commands').value)
        republish_rate_hz = float(p('republish_rate_hz').value)
        if not (math.isfinite(republish_rate_hz) and republish_rate_hz > 0.0):
            raise ValueError(f'republish_rate_hz must be > 0, got {republish_rate_hz}')

        limits = GateLimits(
            max_vx=float(p('max_vx').value),
            max_yaw_rate=float(p('max_yaw_rate').value),
            reject_vx_above=float(p('reject_vx_above').value),
            reject_yaw_rate_above=float(p('reject_yaw_rate_above').value),
            command_timeout_s=float(p('cmd_velocity_timeout_s').value),
            allow_combined=bool(p('allow_combined_motion').value),
        )
        # Raises ValueError on nonsensical limits: the bridge refuses to start.
        self._gate = MotionGate(limits)
        cmd_z = float(p('cmd_z').value)
        self._rotate_yaw_rate = min(abs(cmd_z), limits.max_yaw_rate)
        if self._rotate_yaw_rate != abs(cmd_z):
            self.get_logger().warn(
                f'cmd_z={cmd_z} exceeds max_yaw_rate; rotations use '
                f'{self._rotate_yaw_rate} rad/s and deg_per_sec is no longer calibrated'
            )

        # Monotonic clock: the Jetson has no RTC and its wall clock is set by
        # hand mid-session, which must not stretch or skip the watchdog.
        self._now = time.monotonic

        # -- Publishers --
        self._sport_topic = DRY_RUN_SPORT_TOPIC if self._dry_run else SPORT_TOPIC
        self._sport_pub = self.create_publisher(Request, self._sport_topic, 10)
        self._audio_pub = self.create_publisher(Request, '/api/audiohub/request', 10)
        self._status_pub = self.create_publisher(String, '/come_here/bridge_status', 10)

        # -- Runtime state --
        self._sport_lock = threading.Lock()
        self._last_was_zero = True  # suppress a StopMove at startup
        self._warn_times = {}

        # Rotation thread control: incrementing generation invalidates older rotations.
        self._rotate_generation = 0
        self._rotate_lock = threading.Lock()
        self._rotate_cancel = threading.Event()
        self._rotate_closed_loop = bool(p('rotate_closed_loop').value)
        self._odom_max_age_s = float(p('odom_max_age_s').value)
        self._rotate_deadband_rad = float(p('rotate_deadband_rad').value)
        self._rotate_timeout_s = float(p('rotate_timeout_s').value)
        self._rotate_prefer_ccw_beyond_rad = float(p('rotate_prefer_ccw_beyond_rad').value)
        self._odom_lock = threading.Lock()
        self._odom_yaw = None
        self._odom_stamp_s = None
        self._rotate_result_pub = self.create_publisher(String, '/come_here/rotate_result', 10)
        if self._enable_rotate and self._rotate_closed_loop:
            # Best effort matches the GO2's own publishers whatever their reliability.
            self.create_subscription(
                Odometry, str(p('odom_topic').value), self._odom_cb, qos_profile_sensor_data
            )

        # Audio thread: drop new cmd_say if a previous playback is still streaming.
        self._audio_busy = threading.Event()

        # Motion mode: refuse motion until CheckMode reports the required mode.
        self._motion_mode = None
        self._mode_verified = False
        self._mode_started_s = self._now()
        self._mode_last_request_s = -math.inf
        if self._required_mode:
            self._gate.inhibit('motion_mode_unverified')
            self._mode_pub = self.create_publisher(
                Request, MOTION_SWITCHER_REQUEST_TOPIC, 10
            )
            self.create_subscription(
                Response, MOTION_SWITCHER_RESPONSE_TOPIC, self._mode_response_cb, 10
            )
            self._mode_timer = self.create_timer(0.5, self._mode_check_tick)

        # -- Subscribers --
        self.create_subscription(
            Float64MultiArray, '/come_here/cmd_velocity', self._velocity_cb, 10
        )
        self.create_subscription(Bool, '/come_here/estop', self._estop_cb, 10)
        self.create_subscription(Float64, '/come_here/cmd_rotate', self._rotate_cb, 10)
        self.create_subscription(Bool, '/come_here/cmd_sit', self._sit_cb, 10)
        self.create_subscription(Bool, '/come_here/cmd_stand', self._stand_cb, 10)
        self.create_subscription(String, '/come_here/cmd_say', self._say_cb, 10)

        self._velocity_timer = self.create_timer(
            1.0 / republish_rate_hz, self._velocity_tick
        )
        self._status_timer = self.create_timer(1.0, self._publish_status)

        if self._dry_run:
            self.get_logger().warn(
                f'DRY RUN: Sport API requests go to {self._sport_topic}; the robot will not move'
            )
        self.get_logger().info(
            f'go2_bridge_node started: sport_topic={self._sport_topic} '
            f'require_motion_mode={self._required_mode or "off"} '
            f'max_vx={limits.max_vx} max_yaw_rate={limits.max_yaw_rate} '
            f'reject_above=({limits.reject_vx_above}, {limits.reject_yaw_rate_above}) '
            f'allow_combined={limits.allow_combined} '
            f'timeout={limits.command_timeout_s}s republish={republish_rate_hz}Hz '
            f'rotate={self._enable_rotate} posture={self._enable_posture}'
        )

    # -- Sport API publishing --

    def _publish_move(self, vx: float, yaw_rate: float) -> None:
        with self._sport_lock:
            self._sport_pub.publish(
                make_req(self._move_api_id, {'x': vx, 'y': 0.0, 'z': yaw_rate})
            )
            self._last_was_zero = False

    def _publish_stop(self, force: bool = False) -> None:
        """StopMove once per stop transition, or unconditionally with force."""
        with self._sport_lock:
            if force or not self._last_was_zero:
                self._sport_pub.publish(make_req(self._stop_move_api_id))
                self._last_was_zero = True

    def _warn_throttled(self, key: str, text: str, period_s: float = 1.0) -> None:
        now = self._now()
        if now - self._warn_times.get(key, -math.inf) >= period_s:
            self._warn_times[key] = now
            self.get_logger().warn(text)

    def _apply(self, decision, source: str) -> None:
        if decision.action == MOVE:
            self._publish_move(decision.vx, decision.yaw_rate)
            if decision.reason == 'clamped':
                self._warn_throttled(
                    'clamped',
                    f'{source}: command clamped to vx={decision.vx:.2f} '
                    f'yaw_rate={decision.yaw_rate:.2f}',
                )
        elif decision.action == STOP:
            self._publish_stop()
            if decision.reason != 'zero':
                self._warn_throttled(
                    f'stop:{decision.reason}', f'{source}: stop ({decision.reason})'
                )

    def _preempt_rotation(self) -> None:
        with self._rotate_lock:
            self._rotate_generation += 1
            self._rotate_cancel.set()
            self._rotate_cancel = threading.Event()

    def _publish_status(self) -> None:
        status = {
            'dry_run': self._dry_run,
            'sport_topic': self._sport_topic,
            'estopped': self._gate.estopped,
            'rearm_required': self._gate.rearm_required,
            'inhibited': self._gate.inhibited,
            'inhibit_reason': self._gate.inhibit_reason,
            'required_motion_mode': self._required_mode or None,
            'motion_mode': self._motion_mode,
            'motion_mode_verified': self._mode_verified,
            'moving': self._gate.active,
            'odom_age_s': self._odom_age_s(),
        }
        msg = String()
        msg.data = json.dumps(status)
        self._status_pub.publish(msg)

    # -- motion mode (read-only CheckMode) --

    def _mode_check_tick(self) -> None:
        now = self._now()
        interval = self._mode_recheck_s if self._mode_verified else 1.0
        if now - self._mode_last_request_s >= interval:
            self._mode_last_request_s = now
            self._mode_pub.publish(make_req(MOTION_SWITCHER_CHECK_MODE_API_ID))
        if (not self._mode_verified
                and now - self._mode_started_s > self._mode_timeout_s
                and now - self._warn_times.get('mode_timeout', -math.inf) >= 5.0):
            self._warn_times['mode_timeout'] = now
            self.get_logger().error(
                f'REFUSING MOTION: no motion_switcher CheckMode reply within '
                f'{self._mode_timeout_s:.0f}s (required {self._required_mode!r})'
            )

    def _mode_response_cb(self, msg: Response) -> None:
        if msg.header.identity.api_id != MOTION_SWITCHER_CHECK_MODE_API_ID:
            return
        if msg.header.status.code != 0:
            self._warn_throttled(
                'mode_status', f'CheckMode returned status {msg.header.status.code}', 5.0
            )
            return
        name = parse_mode_response(msg.data)
        if name is None:
            # An unreadable reply proves nothing either way: keep the current
            # state rather than stopping a trial on a garbled message.
            self._warn_throttled('mode_parse', f'Ignoring unreadable CheckMode reply: {msg.data!r}', 5.0)
            return
        ok, text = mode_verdict(name, self._required_mode)
        self._motion_mode = name
        if ok:
            if not self._mode_verified or self._gate.inhibited:
                self.get_logger().info(f'{text}: motion enabled')
            self._mode_verified = True
            self._gate.inhibit(None)
            return
        newly_inhibited = not self._gate.inhibited or self._mode_verified
        self._mode_verified = False
        self._gate.inhibit('motion_mode')
        self._preempt_rotation()
        self._publish_stop(force=newly_inhibited)
        self.get_logger().error(f'REFUSING MOTION: {text}')

    # -- cmd_velocity --

    def _velocity_cb(self, msg: Float64MultiArray) -> None:
        decision = self._gate.on_command(list(msg.data), self._now())
        if decision.action == NONE:
            self._warn_throttled('estopped', 'cmd_velocity ignored: e-stop engaged')
            return
        # A velocity command owns the gait: cancel any in-flight rotation worker
        # so its Move(0, 0, z) publishes cannot fight this command.
        self._preempt_rotation()
        self._apply(decision, 'cmd_velocity')

    def _velocity_tick(self) -> None:
        """Republish the armed command, or stop once the watchdog trips."""
        self._apply(self._gate.on_tick(self._now()), 'watchdog')

    # -- estop --

    def _estop_cb(self, msg: Bool) -> None:
        if msg.data:
            newly_engaged = not self._gate.estopped
            self._gate.engage_estop()
            self._preempt_rotation()
            self._publish_stop(force=True)
            if newly_engaged:
                self.get_logger().error(
                    'ESTOP ENGAGED: StopMove sent, all motion blocked until '
                    '/come_here/estop false'
                )
                self._publish_status()
        elif self._gate.estopped:
            self._gate.release_estop()
            self.get_logger().warn(
                'ESTOP RELEASED: motion stays blocked until a zero cmd_velocity re-arms it'
            )
            self._publish_status()

    # -- cmd_rotate (experimental TURN_TO_SOUND path) --

    def _rotate_cb(self, msg: Float64) -> None:
        if not self._enable_rotate:
            self._warn_throttled('rotate_disabled', 'cmd_rotate ignored: disabled')
            return
        if self._gate.estopped or self._gate.rearm_required or self._gate.inhibited:
            self._warn_throttled('rotate_blocked', 'cmd_rotate ignored: motion blocked')
            return
        target_rad = float(msg.data)
        if not math.isfinite(target_rad) or abs(target_rad) > math.pi:
            self._warn_throttled('rotate_bad', f'cmd_rotate ignored: invalid {target_rad}')
            return
        now = self._now()
        start_yaw = None
        if self._rotate_closed_loop:
            yaw, stamp = self._latest_yaw()
            if yaw is not None and now - stamp <= self._odom_max_age_s:
                start_yaw = yaw
            else:
                self.get_logger().warn(
                    'cmd_rotate: no fresh odometry, falling back to the timed turn '
                    f'({self._deg_per_sec:.0f} deg/s calibration)')
        controller = RotateController(
            target_rad, self._rotate_yaw_rate, now, start_yaw,
            deadband_rad=self._rotate_deadband_rad,
            timeout_s=self._rotate_timeout_s,
            odom_max_age_s=self._odom_max_age_s,
            fallback_deg_per_sec=self._deg_per_sec,
            prefer_ccw_beyond_rad=self._rotate_prefer_ccw_beyond_rad,
        )
        # The rotation replaces any armed velocity command.
        self._gate.disarm('rotate')

        with self._rotate_lock:
            self._rotate_generation += 1
            my_gen = self._rotate_generation
            self._rotate_cancel.set()
            self._rotate_cancel = threading.Event()
            cancel_event = self._rotate_cancel
            threading.Thread(
                target=self._rotate_worker,
                args=(my_gen, controller, target_rad, cancel_event),
                daemon=True,
            ).start()

        self.get_logger().info(
            f'cmd_rotate: {target_rad:+.2f} rad, '
            f'{"closed loop" if controller.closed_loop else "timed"} '
            f'{"left" if controller.sign > 0 else "right"} '
            f'{math.degrees(controller.magnitude_rad):.0f} deg at '
            f'{self._rotate_yaw_rate:.2f} rad/s, gen={my_gen}'
        )

    def _odom_cb(self, msg) -> None:
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if not math.isfinite(yaw):
            return
        with self._odom_lock:
            self._odom_yaw = yaw
            self._odom_stamp_s = self._now()

    def _latest_yaw(self):
        with self._odom_lock:
            return self._odom_yaw, self._odom_stamp_s

    def _odom_age_s(self):
        _, stamp = self._latest_yaw()
        return None if stamp is None else round(self._now() - stamp, 3)

    def _rotate_worker(self, generation, controller, target_rad, cancel_event):
        """Drive the RotateController at 20 Hz, then StopMove. Abort on cancel."""
        start = time.monotonic()
        while True:
            if cancel_event.is_set() or self._gate.estopped or self._gate.inhibited:
                # A newer command preempted us, or motion was blocked; whoever
                # did that owns the robot's state, so no StopMove here.
                return
            yaw, stamp = self._latest_yaw()
            step = controller.step(self._now(), yaw, stamp)
            if step.done:
                break
            self._publish_move(0.0, step.yaw_rate)
            time.sleep(0.05)

        with self._rotate_lock:
            if generation != self._rotate_generation:
                return
        self._publish_stop(force=True)
        result = {
            'target_rad': round(target_rad, 3),
            'mode': 'closed_loop' if controller.closed_loop else 'timed',
            'reason': step.reason,
            'turned_rad': round(controller.turned_rad, 3) if controller.closed_loop else None,
            'seconds': round(time.monotonic() - start, 2),
        }
        msg = String()
        msg.data = json.dumps(result)
        self._rotate_result_pub.publish(msg)
        log = self.get_logger().info if step.reason in ('reached', 'timed', 'overshoot') \
            else self.get_logger().warn
        log(f'rotate gen={generation} {step.reason}: target {target_rad:+.2f} rad, '
            f'turned {result["turned_rad"]} rad in {result["seconds"]}s ({result["mode"]})')

    # -- cmd_sit / cmd_stand (optional post-arrival sequence) --

    def _posture_allowed(self, label: str) -> bool:
        if not self._enable_posture:
            self._warn_throttled('posture_disabled', f'cmd_{label} ignored: disabled')
            return False
        if self._gate.estopped or self._gate.inhibited:
            self._warn_throttled('posture_blocked', f'cmd_{label} ignored: motion blocked')
            return False
        return True

    def _sit_cb(self, msg: Bool) -> None:
        if not msg.data or not self._posture_allowed('sit'):
            return
        self._gate.disarm('sit')
        self._preempt_rotation()
        # StopMove now, Sit 0.5 s later: Sit sent while the trot is still
        # decelerating is silently dropped (hardware, 2026-04-24).
        self._publish_stop(force=True)
        self.get_logger().info(f'cmd_sit: stopping, will sit in 0.5s (api={self._sit_api_id})')
        threading.Thread(
            target=self._deferred_sport_call,
            args=(self._sit_api_id, 0.5, 'sit'),
            daemon=True,
        ).start()

    def _stand_cb(self, msg: Bool) -> None:
        if not msg.data or not self._posture_allowed('stand'):
            return
        self._gate.disarm('stand')
        self.get_logger().info(f'cmd_stand: api_id={self._stand_api_id}')
        with self._sport_lock:
            self._sport_pub.publish(make_req(self._stand_api_id))

    def _deferred_sport_call(self, api_id: int, delay_s: float, label: str) -> None:
        time.sleep(delay_s)
        if self._gate.estopped or self._gate.inhibited:
            self.get_logger().warn(f'cmd_{label} (deferred) dropped: motion blocked')
            return
        self.get_logger().info(f'cmd_{label} (deferred): api_id={api_id}')
        with self._sport_lock:
            self._sport_pub.publish(make_req(api_id))

    # -- cmd_say --

    def _say_cb(self, msg: String) -> None:
        phrase = msg.data or ''
        if not phrase:
            return
        filename = phrase.strip().lower().replace(' ', '_') + '.wav'
        wav_path = os.path.join(self._wav_dir, filename)

        if not os.path.isfile(wav_path):
            self.get_logger().warn(
                f'cmd_say: WAV not found for "{phrase}" at {wav_path}, dropping'
            )
            return

        if self._audio_busy.is_set():
            self.get_logger().warn(
                f'cmd_say: previous playback still in flight, dropping "{phrase}"'
            )
            return

        # Playback runs on its own thread so it never delays motion commands.
        self._audio_busy.set()
        threading.Thread(
            target=self._play_wav_worker, args=(wav_path, phrase), daemon=True
        ).start()

    def _play_wav_worker(self, wav_path: str, phrase: str) -> None:
        """Stream a WAV file to the GO2 speaker via audiohub (start/chunk/end)."""
        try:
            with open(wav_path, 'rb') as f:
                wav_data = f.read()
            b64 = base64.b64encode(wav_data).decode('utf-8')
            chunk_size = self._wav_chunk_size
            chunks = [b64[i:i + chunk_size] for i in range(0, len(b64), chunk_size)]

            self.get_logger().info(
                f'cmd_say: streaming "{phrase}" ({len(wav_data)} bytes, '
                f'{len(chunks)} chunks)'
            )

            self._audio_pub.publish(make_req(4001))
            time.sleep(0.1)
            for i, chunk in enumerate(chunks):
                payload = {
                    'current_block_index': i + 1,
                    'total_block_number': len(chunks),
                    'block_content': chunk,
                }
                self._audio_pub.publish(make_req(4003, payload))
                time.sleep(self._wav_chunk_delay_s)
            # Let the last chunk play out, then end the session.
            time.sleep(1.5)
            self._audio_pub.publish(make_req(4002))
        except Exception as exc:
            self.get_logger().error(f'cmd_say: playback error for "{phrase}": {exc}')
        finally:
            self._audio_busy.clear()

    def destroy_node(self):
        """Publish a final StopMove so the GO2 never holds its last setpoint.

        run_node() guarantees this runs while the ROS context is still valid.
        """
        try:
            self._gate.disarm('shutdown')
            self._preempt_rotation()
            self._publish_stop(force=True)
        except Exception:  # noqa: BLE001 - shutdown must never raise
            pass
        super().destroy_node()


def main(args=None):
    run_node(Go2BridgeNode, args=args)


if __name__ == '__main__':
    main()
