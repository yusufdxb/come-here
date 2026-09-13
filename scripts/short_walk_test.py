#!/usr/bin/env python3
"""Lab stage D: one short straight walk through the real bridge. THIS MOVES THE ROBOT.

Run only with the Unitree remote in hand, 3 m clear in front of the robot, the
live launch running (dry_run:=false), and nobody in the camera view.

Sends [0, 0] (the bridge needs a zero before it re-arms), waits for Enter,
then [vx, 0] at 10 Hz for --seconds, then [0, 0]. If this script dies, the
bridge watchdog still stops the robot 0.5 s after the last command.

    python3 scripts/short_walk_test.py                  # 0.6 m/s for 1.5 s, about 0.9 m
"""

import argparse
import sys
import time


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--vx', type=float, default=0.6, help='0.5 to 0.7 m/s')
    parser.add_argument('--seconds', type=float, default=1.5, help='0.5 to 3.0 s')
    args = parser.parse_args(argv)
    if not 0.5 <= args.vx <= 0.7 or not 0.5 <= args.seconds <= 3.0:
        print('Refusing: vx must be 0.5-0.7 m/s and seconds 0.5-3.0.')
        return 2

    import rclpy
    from std_msgs.msg import Float64MultiArray

    rclpy.init()
    node = rclpy.create_node('come_here_short_walk_test')
    pub = node.create_publisher(Float64MultiArray, '/come_here/cmd_velocity', 10)

    def send(vx):
        msg = Float64MultiArray()
        msg.data = [float(vx), 0.0]
        pub.publish(msg)

    try:
        deadline = time.monotonic() + 10.0
        while node.count_subscribers('/come_here/cmd_velocity') < 1:
            if time.monotonic() > deadline:
                print('No subscriber on /come_here/cmd_velocity: is the live launch running?')
                return 1
            rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(1.0)
        for _ in range(3):
            send(0.0)
            time.sleep(0.1)
        input(f'Remote in hand, 3 m clear. Enter walks {args.vx} m/s for {args.seconds} s '
              '(Ctrl+C aborts): ')
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            send(args.vx)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print('aborted')
    finally:
        for _ in range(5):
            send(0.0)
            time.sleep(0.05)
        node.destroy_node()
        rclpy.try_shutdown()
    print('done: zero command sent')
    return 0


if __name__ == '__main__':
    sys.exit(main())
