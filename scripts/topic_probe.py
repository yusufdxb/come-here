#!/usr/bin/env python3
"""Read-only topic probes for the demo preflight. Never publishes anything.

    topic_probe.py rate TOPIC TYPE SECONDS   # prints the measured rate in Hz
    topic_probe.py json TOPIC TIMEOUT        # prints one std_msgs/String payload

Running as a separate process is the point: it proves an operator shell can
actually see the launched nodes, not just that the nodes exist.
"""

import sys
import time

TYPES = {
    'image': ('sensor_msgs.msg', 'Image', True),
    'cloud': ('sensor_msgs.msg', 'PointCloud2', True),
    'odom': ('nav_msgs.msg', 'Odometry', True),
    'array': ('std_msgs.msg', 'Float64MultiArray', False),
    'string': ('std_msgs.msg', 'String', False),
}


def _node(name):
    import rclpy
    rclpy.init()
    return rclpy, rclpy.create_node(name)


def rate(topic, kind, seconds):
    import importlib
    from rclpy.qos import qos_profile_sensor_data
    module, cls, sensor = TYPES[kind]
    msg_type = getattr(importlib.import_module(module), cls)
    rclpy, node = _node('come_here_rate_probe')
    count = [0]
    first = [None]

    def on_msg(_msg):
        if first[0] is None:
            first[0] = time.monotonic()
        count[0] += 1

    node.create_subscription(msg_type, topic, on_msg, qos_profile_sensor_data if sensor else 10)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.try_shutdown()
    if count[0] < 2:
        print('0.0')
        return 0
    print(f'{(count[0] - 1) / max(1e-6, time.monotonic() - first[0]):.2f}')
    return 0


def grab_json(topic, timeout):
    from std_msgs.msg import String
    rclpy, node = _node('come_here_json_probe')
    got = []
    node.create_subscription(String, topic, lambda m: got.append(m.data), 10)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not got:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.try_shutdown()
    if not got:
        return 1
    print(got[0])
    return 0


def main(argv):
    if len(argv) == 4 and argv[0] == 'rate':
        return rate(argv[1], argv[2], float(argv[3]))
    if len(argv) == 3 and argv[0] == 'json':
        return grab_json(argv[1], float(argv[2]))
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
