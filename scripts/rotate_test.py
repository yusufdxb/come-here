#!/usr/bin/env python3
"""Lab stage E: one closed-loop turn in place through the real bridge. THIS MOVES THE ROBOT.

Run only with the Unitree remote in hand, 1.5 m clear all round, the live
launch running (dry_run:=false) and nobody speaking. Sends one
/come_here/cmd_rotate, watches /utlidar/robot_odom, and prints the measured
yaw change against the target. Ctrl+C publishes a zero cmd_velocity, which
preempts the turn and stops the robot.

    python3 scripts/rotate_test.py --deg 90      # turn left a quarter turn
    python3 scripts/rotate_test.py --deg -90     # turn right
    python3 scripts/rotate_test.py --deg 170     # near-behind: turns left the long way
"""

import argparse
import json
import math
import signal
import sys
import time


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def _yaw(msg):
    q = msg.pose.pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--deg', type=float, default=90.0, help='20 to 180, + = left')
    parser.add_argument('--timeout', type=float, default=8.0)
    args = parser.parse_args(argv)
    if not 20.0 <= abs(args.deg) <= 180.0:
        print('Refusing: |deg| must be 20-180.')
        return 2

    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import Float64, Float64MultiArray, String

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _interrupt)
    node = rclpy.create_node('come_here_rotate_test')
    vel_pub = node.create_publisher(Float64MultiArray, '/come_here/cmd_velocity', 10)
    rot_pub = node.create_publisher(Float64, '/come_here/cmd_rotate', 10)
    state = {'yaw': None, 'result': None, 'samples': 0}

    def odom_cb(msg):
        state['yaw'] = _yaw(msg)
        state['samples'] += 1

    def result_cb(msg):
        state['result'] = json.loads(msg.data)

    node.create_subscription(Odometry, '/utlidar/robot_odom', odom_cb, qos_profile_sensor_data)
    node.create_subscription(String, '/come_here/rotate_result', result_cb, 10)

    def zero():
        msg = Float64MultiArray()
        msg.data = [0.0, 0.0]
        vel_pub.publish(msg)

    try:
        deadline = time.monotonic() + 10.0
        while (node.count_subscribers('/come_here/cmd_rotate') < 1
               or state['samples'] < 5):
            if time.monotonic() > deadline:
                print('need the live launch (cmd_rotate subscriber) and /utlidar/robot_odom')
                return 1
            rclpy.spin_once(node, timeout_sec=0.1)
        yaw0 = state['yaw']
        input(f'Remote in hand, area clear. Enter turns {args.deg:+.0f} deg '
              f'(yaw now {math.degrees(yaw0):+.1f}); Ctrl+C aborts: ')
        msg = Float64()
        msg.data = math.radians(args.deg)
        t0 = time.monotonic()
        rot_pub.publish(msg)
        while state['result'] is None and time.monotonic() - t0 < args.timeout:
            rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.5)
        rclpy.spin_once(node, timeout_sec=0.1)
        turned = _wrap(state['yaw'] - yaw0)
        target = math.radians(args.deg)
        print(f'target {args.deg:+.1f} deg, odometry turned {math.degrees(turned):+.1f} deg, '
              f'error {math.degrees(_wrap(turned - target)):+.1f} deg, '
              f'{time.monotonic() - t0:.1f} s')
        print(f'bridge result: {state["result"]}')
        if state['result'] is None:
            print('no /come_here/rotate_result within the timeout: check the bridge log')
    except (KeyboardInterrupt, EOFError):
        print('aborted')
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        for _ in range(5):
            zero()
            time.sleep(0.05)
        node.destroy_node()
        rclpy.try_shutdown()
    print('done: zero command sent')
    return 0


if __name__ == '__main__':
    sys.exit(main())
