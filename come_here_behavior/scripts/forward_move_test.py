"""Forward-only direct-Move test for GO2.

Bypasses come-here FSM. Sends Sport API Move(vx, 0, 0) at 10 Hz for --duration
seconds, then StopMove. Isolates gait quality from the APPROACH control loop:

  smooth gait => APPROACH oscillation is control-loop (yaw gain / stale bearing)
  shaky gait  => firmware / gait layer

No come_here_* node should be running. Robot should be standing with >=1.5 m
clear forward space. Ctrl+C triggers StopMove and exits.
"""

import argparse
import datetime
import json
import random
import sys
import time

import rclpy
from unitree_api.msg import Request

MOVE_API_ID = 1008
STOP_MOVE_API_ID = 1003
STAND_API_ID = 1002  # BalanceStand


def make_req(api_id, params=None):
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
        msg.parameter = json.dumps(params)
    else:
        msg.parameter = ""
    msg.binary = []
    return msg


def log(tag, msg):
    print(f"[{time.time():.3f}] {tag}: {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vx", type=float, default=0.2,
                        help="forward velocity m/s (default 0.2)")
    parser.add_argument("--duration", type=float, default=1.5,
                        help="seconds to hold vx (default 1.5)")
    parser.add_argument("--rate", type=float, default=10.0,
                        help="publish rate Hz (default 10)")
    parser.add_argument("--stand", action="store_true",
                        help="send BalanceStand + wait 2s before Move")
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("forward_move_test")
    pub = node.create_publisher(Request, "/api/sport/request", 10)

    log("init", f"vx={args.vx} duration={args.duration}s rate={args.rate}Hz stand={args.stand}")
    log("wait", "0.5s for DDS discovery")
    time.sleep(0.5)

    if args.stand:
        log("stand", f"BalanceStand api_id={STAND_API_ID}")
        pub.publish(make_req(STAND_API_ID))
        time.sleep(2.0)

    log("move_start", f"Move(vx={args.vx}, 0, 0) for {args.duration}s")
    period = 1.0 / args.rate
    ticks = int(args.duration * args.rate)
    try:
        for i in range(ticks):
            pub.publish(make_req(MOVE_API_ID, {"x": args.vx, "y": 0.0, "z": 0.0}))
            time.sleep(period)
    except KeyboardInterrupt:
        log("interrupt", "Ctrl+C — stopping")

    log("stop", f"StopMove api_id={STOP_MOVE_API_ID}")
    pub.publish(make_req(STOP_MOVE_API_ID))
    time.sleep(0.3)
    pub.publish(make_req(STOP_MOVE_API_ID))  # belt-and-suspenders
    time.sleep(0.2)

    node.destroy_node()
    rclpy.shutdown()
    log("done", "exit")


if __name__ == "__main__":
    sys.exit(main())
