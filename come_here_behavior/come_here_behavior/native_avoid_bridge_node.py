"""Come Here ANY bridge: the legacy GO2 bridge with the native-avoidance motion backend.

``Go2BridgeNode`` (legacy, unchanged) keeps every behavior it already has:
latched e-stop, remote-stick manual override, CheckMode refusal, watchdog,
closed-loop rotate, deferred Sit, speech. This subclass changes only WHO the
motion goes to and WHAT a command may contain:

  * every Move / StopMove, including the rotate worker's, goes through one
    ``NativeAvoidBackend`` (``native_avoid_backend``: sport_freeavoid or
    obstacles_avoid). It is the only motion owner in the ANY launch; the
    legacy launch never instantiates it.
  * ``/come_here/cmd_velocity`` carries ``[vx, vy, yaw_rate]`` and passes a
    ``NativeMotionGate``: a 2-element legacy command is rejected as malformed,
    lateral and combined motion each need their own flag (default off).
  * no motion is published unless the backend reports ENABLED; an enable
    failure or timeout latches FAILED (inhibit) until the node restarts.
  * live motion (dry_run false) additionally needs
    ``native_live_motion_cleared: true``; otherwise nothing is enabled and
    every command stops (hardware NO-GO until the validation ladder passes).
  * e-stop / manual override / Sit: stop, then release obstacles_avoid API
    remote-command ownership at once, so the remote gets its authority back.
  * shutdown: stop, release API control, restore / disable avoidance.
  * odometry older than ``native_odom_max_age_s`` while moving: stop.

Dry run: Sport and obstacles_avoid requests go to /come_here/dry_run/*, and
the enable replies are SIMULATED (enable_result 'dry_run_simulated'). A dry run
proves routing, ownership and sequencing, never obstacle-avoidance behavior.

Extra parameters (the legacy ones are documented in go2_bridge_node.py):
  native_avoid_backend             (str,  'sport_freeavoid')
  native_live_motion_cleared       (bool, False) required for live motion
  native_allow_lateral             (bool, False) accept vy
  native_allow_combined            (bool, False) accept translation + yaw together
  native_max_vy / native_reject_vy_above (float, 0.3 / 1.0)
  native_response_timeout_s        (float, 1.5)  per enable step
  native_use_remote_command_from_api (bool, True) obstacles_avoid only
  native_restore_switch_on_release (bool, True)  obstacles_avoid: SwitchSet(initial)
  native_disable_freeavoid_on_release (bool, True) freeavoid: FreeAvoid(false) at exit
  native_stop_with_sport_stopmove  (bool, True)  obstacles_avoid: StopMove after zero Move
  native_odom_max_age_s            (float, 0.5)
"""

import json
import math
import random
import threading
import time

from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from unitree_api.msg import Request, Response

from come_here_behavior.go2_bridge_node import Go2BridgeNode, make_req
from come_here_behavior.native_avoid_backend import (
    DRY_RUN_REQUEST_TOPICS,
    MOVE,
    NONE,
    OBSTACLES_AVOID,
    REQUEST_TOPICS,
    RESPONSE_TOPICS,
    SPORT,
    STOP,
    NativeAvoidBackend,
    NativeGateLimits,
    NativeMotionGate,
)
from come_here_behavior.node_runner import run_node


class NativeAvoidBridgeNode(Go2BridgeNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare_parameter('native_avoid_backend', 'sport_freeavoid')
        self.declare_parameter('native_live_motion_cleared', False)
        self.declare_parameter('native_allow_lateral', False)
        self.declare_parameter('native_allow_combined', False)
        self.declare_parameter('native_max_vy', 0.3)
        self.declare_parameter('native_reject_vy_above', 1.0)
        self.declare_parameter('native_response_timeout_s', 1.5)
        self.declare_parameter('native_use_remote_command_from_api', True)
        self.declare_parameter('native_restore_switch_on_release', True)
        self.declare_parameter('native_disable_freeavoid_on_release', True)
        self.declare_parameter('native_stop_with_sport_stopmove', True)
        self.declare_parameter('native_odom_max_age_s', 0.5)
        p = self.get_parameter

        old = self._gate
        base = old._limits
        limits = NativeGateLimits(
            max_vx=base.max_vx, max_vy=float(p('native_max_vy').value),
            max_yaw_rate=base.max_yaw_rate,
            reject_vx_above=base.reject_vx_above,
            reject_vy_above=float(p('native_reject_vy_above').value),
            reject_yaw_rate_above=base.reject_yaw_rate_above,
            command_timeout_s=base.command_timeout_s,
            allow_combined=bool(p('native_allow_combined').value),
            allow_lateral=bool(p('native_allow_lateral').value),
        )
        # Nothing has spun yet, so no callback has seen the legacy gate.
        gate = NativeMotionGate(limits)
        if old.inhibited:
            gate.inhibit(old.inhibit_reason)
        self._gate = gate

        self._backend = NativeAvoidBackend(
            str(p('native_avoid_backend').value),
            response_timeout_s=float(p('native_response_timeout_s').value),
            use_remote_command_from_api=bool(p('native_use_remote_command_from_api').value),
            restore_switch_on_release=bool(p('native_restore_switch_on_release').value),
            disable_freeavoid_on_release=bool(p('native_disable_freeavoid_on_release').value),
            stop_with_sport_stopmove=bool(p('native_stop_with_sport_stopmove').value),
            simulate_responses=self._dry_run,
        )
        self._backend._next_id = random.randint(1_000_000_000, 1_900_000_000)
        self._live_cleared = bool(p('native_live_motion_cleared').value)
        self._native_odom_max_age_s = float(p('native_odom_max_age_s').value)
        self._flag_lock = threading.Lock()
        self._posture_hold = False
        self._released = False
        self._failed_handled = False

        topics = DRY_RUN_REQUEST_TOPICS if self._dry_run else REQUEST_TOPICS
        self._oa_topic = topics[OBSTACLES_AVOID]
        self._oa_pub = self.create_publisher(Request, self._oa_topic, 10)
        if not self._dry_run:
            for service in (SPORT, OBSTACLES_AVOID):
                self.create_subscription(
                    Response, RESPONSE_TOPICS[service],
                    lambda msg, s=service: self._response_cb(s, msg), 10)
        if not (self._enable_rotate and self._rotate_closed_loop):
            # The legacy bridge subscribes only for closed-loop turns; ANY always
            # needs odometry for its freshness stop.
            self.create_subscription(Odometry, str(p('odom_topic').value), self._odom_cb,
                                     qos_profile_sensor_data)

        if not self._dry_run and not self._live_cleared:
            self._gate.set_inhibit('native_live_motion_not_cleared', True)
            self.get_logger().error(
                'COME HERE ANY LIVE MOTION NOT CLEARED: native avoidance is not enabled and '
                'every motion command stops. Set native_live_motion_cleared:=true only after '
                'the Stage A-D ladder in docs/come_here_any.md has passed.')
        self._native_timer = self.create_timer(0.1, self._native_tick)
        self.get_logger().warn(
            f'COME HERE ANY bridge: backend={self._backend.backend} '
            f'lateral={limits.allow_lateral} combined={limits.allow_combined} '
            f'max_vy={limits.max_vy} oa_topic={self._oa_topic} '
            f'live_cleared={self._live_cleared} dry_run={self._dry_run}')

    # -- publishing through the one motion owner --

    def _send(self, calls, now=None):
        """Publish calls in order; returns the request id of the last one."""
        rid = None
        for call in calls:
            rid = self._backend.new_request_id()
            msg = make_req(call.api_id)
            msg.header.identity.id = rid
            msg.header.policy.noreply = bool(call.noreply)
            msg.parameter = call.parameter_json()
            pub = self._sport_pub if call.service == SPORT else self._oa_pub
            with self._sport_lock:
                pub.publish(msg)
        return rid

    def _pump(self, calls, now):
        """Drive the enable sequence: publish, register, (dry run) simulate the reply."""
        while calls:
            rid = self._send(calls[:1])
            self._backend.on_published(rid, now)
            calls = self._backend.simulated_reply(now) if self._dry_run else []
        self._log_backend()

    def _publish_move(self, vx: float, yaw_rate: float, vy: float = 0.0) -> None:
        # Decide, publish and update the zero flag atomically (as the legacy bridge
        # does under its one lock): a stop racing the rotate worker must see this Move.
        with self._flag_lock:
            calls = self._backend.move(vx, vy, yaw_rate)
            if calls:
                self._send(calls)
                self._last_was_zero = False
                return
        self._publish_stop()

    def _publish_stop(self, force: bool = False) -> None:
        with self._flag_lock:
            if not (force or not self._last_was_zero):
                return
            # After the shutdown release only the known-good Sport StopMove is sent.
            calls = self._backend.stop_calls() if not self._released else [
                c for c in self._backend.stop_calls() if c.service == SPORT]
            self._send(calls)
            self._last_was_zero = True

    def _apply(self, decision, source: str) -> None:
        if decision.action == MOVE:
            self._publish_move(decision.vx, decision.yaw_rate, decision.vy)
            if decision.reason == 'clamped':
                self._warn_throttled(
                    'clamped', f'{source}: command clamped to vx={decision.vx:.2f} '
                    f'vy={decision.vy:.2f} yaw_rate={decision.yaw_rate:.2f}')
        elif decision.action == STOP:
            self._publish_stop()
            if decision.reason != 'zero':
                self._warn_throttled(f'stop:{decision.reason}', f'{source}: stop ({decision.reason})')

    def _odom_fresh(self) -> bool:
        age = self._odom_age_s()
        return age is not None and age <= self._native_odom_max_age_s

    def _motion_blocker(self):
        if not self._backend.enabled:
            return f'native avoidance {self._backend.state}'
        if not self._odom_fresh():
            return f'odometry stale (age {self._odom_age_s()})'
        return None

    def _velocity_cb(self, msg) -> None:
        decision = self._gate.on_command(list(msg.data), self._now())
        if decision.action == NONE:
            self._warn_throttled('estopped', 'cmd_velocity ignored: e-stop engaged')
            return
        self._preempt_rotation()
        if decision.action == MOVE:
            blocker = self._motion_blocker()
            if blocker is not None:
                self._gate.disarm('native_blocked')
                self._publish_stop()
                self._warn_throttled('native_blocked', f'cmd_velocity refused: {blocker}')
                return
        self._apply(decision, 'cmd_velocity')

    def _velocity_tick(self) -> None:
        decision = self._gate.on_tick(self._now())
        if decision.action == MOVE:
            blocker = self._motion_blocker()
            if blocker is not None:
                self._gate.disarm('native_blocked')
                self._publish_stop(force=True)
                self.get_logger().error(f'STOP while moving: {blocker}')
                return
        self._apply(decision, 'watchdog')

    def _rotate_cb(self, msg) -> None:
        blocker = self._motion_blocker()
        if blocker is not None:
            self._warn_throttled('rotate_native', f'cmd_rotate ignored: {blocker}')
            return
        super()._rotate_cb(msg)

    # -- enable / suspend / release --

    def _enable_allowed(self) -> bool:
        if self._gate.estopped or self._posture_hold or self._backend.failed:
            return False
        if not self._dry_run and not self._live_cleared:
            return False
        if self._required_mode and not self._mode_verified:
            return False
        return True

    def _native_tick(self) -> None:
        now = self._now()
        self._pump(self._backend.tick(now), now)
        if self._backend.failed and not self._failed_handled:
            self._failed_handled = True
            self._gate.set_inhibit('native_avoid_failed', True)
            self._preempt_rotation()
            self._publish_stop(force=True)
            self.get_logger().error(
                f'NATIVE AVOID FAILED ({self._backend.status().failure}): motion blocked until '
                'the node restarts; Come Here ANY does not fall back to legacy walking')
            return
        if (self._required_mode and not self._mode_verified
                and self._backend.state in ('enabling', 'enabled')):
            self._suspend('motion mode no longer verified')
            return
        if self._backend.state == 'disabled' and self._enable_allowed():
            self._pump(self._backend.begin_enable(now), now)

    def _response_cb(self, service: str, msg) -> None:
        now = self._now()
        calls = self._backend.on_response(
            service, int(msg.header.identity.id), int(msg.header.identity.api_id),
            int(msg.header.status.code), msg.data, now)
        self._pump(calls, now)

    def _log_backend(self) -> None:
        # One call site per severity: rclpy refuses a severity change at one site.
        for line in self._backend.drain_log():
            if 'FAILED' in line:
                self.get_logger().error(line)
            else:
                self.get_logger().info(line)

    def _suspend(self, why: str) -> None:
        calls = self._backend.suspend()
        self._send(calls)
        self.get_logger().warn(f'native avoid suspended ({why}): '
                               + ', '.join(c.label for c in calls))

    def _estop_cb(self, msg) -> None:
        super()._estop_cb(msg)
        if msg.data:
            self._suspend('e-stop')

    def _sit_cb(self, msg) -> None:
        if msg.data and self._enable_on_sit_allowed():
            self._posture_hold = True
            self._suspend('sit')
        super()._sit_cb(msg)

    def _enable_on_sit_allowed(self) -> bool:
        # Mirror the legacy _posture_allowed: a Sit that will be dropped must not
        # leave the posture hold set.
        return self._enable_posture and not self._gate.estopped and not self._gate.inhibited

    def _stand_cb(self, msg) -> None:
        super()._stand_cb(msg)
        if msg.data and self._enable_posture and not self._gate.estopped:
            self._posture_hold = False

    def _publish_status(self) -> None:
        status = {
            'dry_run': self._dry_run,
            'sport_topic': self._sport_topic,
            'obstacles_avoid_topic': self._oa_topic,
            'estopped': self._gate.estopped,
            'rearm_required': self._gate.rearm_required,
            'inhibited': self._gate.inhibited,
            'inhibit_reason': self._gate.inhibit_reason,
            'required_motion_mode': self._required_mode or None,
            'motion_mode': self._motion_mode,
            'motion_mode_verified': self._mode_verified,
            'moving': self._gate.active,
            'odom_age_s': self._odom_age_s(),
            'mode': 'any',
            'native_live_motion_cleared': self._live_cleared,
            'native_allow_lateral': self._gate.limits.allow_lateral,
            'native_allow_combined': self._gate.limits.allow_combined,
            'posture_hold': self._posture_hold,
        }
        self._log_backend()
        status.update(self._backend.status().as_dict())
        msg = String()
        msg.data = json.dumps(status)
        self._status_pub.publish(msg)

    def destroy_node(self):
        """Stop, release API control, restore avoidance, then the legacy final StopMove."""
        try:
            self._gate.disarm('shutdown')
            self._preempt_rotation()
            calls = self._backend.release_calls()
            for call in calls:
                self._send([call])
                time.sleep(0.05)
            self._released = True
            self.get_logger().warn('native avoid released at shutdown: '
                                   + ', '.join(c.label for c in calls))
        except Exception:  # noqa: BLE001 - shutdown must never raise
            pass
        super().destroy_node()


def main(args=None):
    run_node(NativeAvoidBridgeNode, args=args)


if __name__ == '__main__':
    main()
