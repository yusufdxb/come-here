"""Come Here ANY behavior node: the legacy BehaviorNode running ``ComeHereAnyFsm``.

Everything the legacy node does is inherited (topics, trial log, e-stop,
shutdown stop). Added:

Subscribes:
  <odom_topic>             (nav_msgs/Odometry) robot pose for the caller estimate and the
                           travel / displacement bounds (/utlidar/robot_odom)
  /come_here/bridge_status (std_msgs/String) JSON; the native-avoidance fields are copied
                           into every ANY trial record

Publishes /come_here/cmd_velocity as ``[vx, vy, yaw_rate]`` (three elements: the
legacy bridge rejects it as malformed, so ANY cannot drive the legacy backend).

Parameters: every ``AnyFsmConfig`` field (come_here_any_controller.py), plus
``odom_topic`` and the legacy node parameters.
"""

import json
import math

from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from come_here_behavior.behavior_node import BehaviorNode
from come_here_behavior.come_here_any_controller import AnyFsmConfig, ComeHereAnyFsm
from come_here_behavior.node_runner import run_node

# Bridge status keys copied into the trial record (native_avoid_backend.BackendStatus).
BRIDGE_FIELDS = (
    'native_avoid_backend', 'native_avoid_state', 'native_avoid_enabled',
    'native_avoid_enable_result', 'native_avoid_server_version',
    'native_avoid_api_version_match', 'api_control_taken', 'api_control_released',
    'native_avoid_simulated', 'native_avoid_failure', 'native_live_motion_cleared',
    'native_allow_lateral', 'native_allow_combined', 'dry_run',
)


class ComeHereAnyBehaviorNode(BehaviorNode):
    CONFIG_CLASS = AnyFsmConfig
    FSM_CLASS = ComeHereAnyFsm

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare_parameter('odom_topic', '/utlidar/robot_odom')
        self._bridge_status = {}
        self._bridge_status_at_start = None
        self.create_subscription(
            Odometry, str(self.get_parameter('odom_topic').value), self._odom_cb,
            qos_profile_sensor_data)
        self.create_subscription(String, '/come_here/bridge_status', self._bridge_status_cb, 10)
        self.get_logger().warn(
            f'COME HERE ANY behavior: control_law={self._fsm.config.any_control_law} '
            f'max_travel={self._fsm.config.any_max_travel_m} m '
            f'prediction_ttl={self._fsm.config.any_prediction_ttl_s} s')

    def _odom_cb(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        p = msg.pose.pose.position
        self._apply(self._fsm.on_odom(p.x, p.y, yaw, self._now()))

    def _bridge_status_cb(self, msg: String) -> None:
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if isinstance(status, dict):
            self._bridge_status = status

    def _wake_cb(self, msg: String) -> None:
        started = not self._fsm.trial_active
        super()._wake_cb(msg)
        if started and self._fsm.trial_active:
            self._bridge_status_at_start = dict(self._bridge_status)

    def _apply(self, cmds) -> None:
        if cmds.trial_summary is not None:
            status = self._bridge_status
            start = self._bridge_status_at_start or {}
            cmds.trial_summary.update({k: status.get(k) for k in BRIDGE_FIELDS})
            cmds.trial_summary['native_avoid_enabled_at_wake'] = start.get('native_avoid_enabled')
            cmds.trial_summary['bridge_status_seen'] = bool(status)
            self._bridge_status_at_start = None
        super()._apply(cmds)


def main(args=None):
    run_node(ComeHereAnyBehaviorNode, args=args)


if __name__ == '__main__':
    main()
