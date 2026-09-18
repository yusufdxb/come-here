#!/usr/bin/env python3
"""Lab stage D: one short single-axis motion through the real bridge. THIS MOVES THE ROBOT.

Run only with the Unitree remote in hand, 3 m clear around the robot, the live
launch running (dry_run:=false), and nobody in the camera view.

Sends [0, 0] (the bridge needs a zero before it re-arms), waits for Enter,
sends one axis at 10 Hz for --seconds, then [0, 0]. If this script dies, the
bridge watchdog still stops the robot 0.5 s after the last command.

    python3 scripts/short_walk_test.py                       # forward 0.6 m/s for 1.5 s, about 0.9 m
    python3 scripts/short_walk_test.py --yaw 0.6 --seconds 1.5    # turn left in place
    python3 scripts/short_walk_test.py --yaw -0.6 --seconds 1.5   # turn right in place

The yaw runs check that the ALIGN rate (approach_ccw_yaw / approach_cw_yaw)
actually makes the feet step rather than only twisting the body.
"""

import argparse
import signal
import sys
import time


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--vx', type=float, default=0.6, help='forward, 0.5 to 0.7 m/s')
    parser.add_argument('--yaw', type=float, default=None,
                        help='turn in place instead, |yaw| 0.5 to 1.0 rad/s (+ = left)')
    parser.add_argument('--seconds', type=float, default=1.5, help='0.5 to 3.0 s')
    args = parser.parse_args(argv)
    if not 0.5 <= args.seconds <= 3.0:
        print('Refusing: seconds must be 0.5-3.0.')
        return 2
    if args.yaw is None:
        if not 0.5 <= args.vx <= 0.7:
            print('Refusing: vx must be 0.5-0.7 m/s.')
            return 2
        command, label = [args.vx, 0.0], f'forward {args.vx} m/s'
    else:
        if not 0.5 <= abs(args.yaw) <= 1.0:
            print('Refusing: |yaw| must be 0.5-1.0 rad/s.')
            return 2
        command, label = [0.0, args.yaw], f'turn {args.yaw:+} rad/s'

    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import Float64MultiArray

    # rclpy's own handlers would shut ROS down on Ctrl+C before the final
    # zero command could be published.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _interrupt)
    node = rclpy.create_node('come_here_short_walk_test')
    pub = node.create_publisher(Float64MultiArray, '/come_here/cmd_velocity', 10)

    def send(values):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in values]
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
            send([0.0, 0.0])
            time.sleep(0.1)
        input(f'Remote in hand, area clear. Enter sends {label} for {args.seconds} s '
              '(Ctrl+C aborts): ')
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            send(command)
            time.sleep(0.1)
    except (KeyboardInterrupt, EOFError):
        print('aborted')
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        for _ in range(5):
            send([0.0, 0.0])
            time.sleep(0.05)
        node.destroy_node()
        rclpy.try_shutdown()
    print('done: zero command sent')
    return 0


if __name__ == '__main__':
    sys.exit(main())
