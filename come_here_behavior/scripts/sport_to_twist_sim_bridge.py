#!/usr/bin/env python3
"""Sim bridge: GO2 Sport API requests -> geometry_msgs/Twist for go2_omniverse.

come-here's go2_bridge_node speaks the Unitree Sport API on /api/sport/request
(unitree_api/msg/Request). The go2_omniverse Isaac Sim subscribes to
/robot0/cmd_vel (geometry_msgs/Twist) and feeds it to the RL locomotion
policy. This node translates between them so come-here can drive the simulated
GO2 in a closed loop:

  api_id 1008 (Move):     parameter JSON {x, y, z} -> Twist(linear.x=x,
                          linear.y=y, angular.z=z)
  api_id 1003 (StopMove): -> zero Twist
  others (Sit/Stand/audiohub): ignored (no sim analogue is needed for the
                          motion-safety scenarios this harness verifies)

This is a simulation / test utility, NOT part of the come-here runtime. Run it
alongside go2_bridge_node when testing against go2_omniverse:

  ros2 run come_here_behavior ...   # behavior_node + go2_bridge_node
  python3 scripts/sport_to_twist_sim_bridge.py
"""

import json

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from unitree_api.msg import Request

MOVE_API_ID = 1008
STOP_MOVE_API_ID = 1003


class SportToTwistSimBridge(Node):
    def __init__(self):
        super().__init__('sport_to_twist_sim_bridge')
        self.declare_parameter('cmd_vel_topic', '/robot0/cmd_vel')
        self.declare_parameter('sport_topic', '/api/sport/request')
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        sport_topic = self.get_parameter('sport_topic').value

        self._pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.create_subscription(Request, sport_topic, self._on_request, 10)
        self.get_logger().info(
            f'sport_to_twist: {sport_topic} (Sport API) -> {cmd_vel_topic} (Twist)'
        )

    def _on_request(self, msg: Request) -> None:
        api_id = msg.header.identity.api_id
        twist = Twist()
        if api_id == MOVE_API_ID:
            try:
                params = json.loads(msg.parameter) if msg.parameter else {}
            except (json.JSONDecodeError, TypeError):
                self.get_logger().warn(
                    f'Move with unparseable parameter: {msg.parameter!r}'
                )
                return
            twist.linear.x = float(params.get('x', 0.0))
            twist.linear.y = float(params.get('y', 0.0))
            twist.angular.z = float(params.get('z', 0.0))
            self._pub.publish(twist)
        elif api_id == STOP_MOVE_API_ID:
            self._pub.publish(twist)  # all-zero Twist
        # other api_ids have no sim analogue for these scenarios; ignore.


def main(args=None):
    rclpy.init(args=args)
    node = SportToTwistSimBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
