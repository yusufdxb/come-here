"""GO2 motion-switcher mode check: read-only, and never on the Sport topic.

CheckMode is api_id 1001 on /api/motion_switcher/request. The SAME number on
/api/sport/request is Damp, which drops a standing robot where it stands, so
the topics are named constants here and the bridge refuses any Sport api id of
1001. Only 'mcf' and 'ai' are valid modes on the GO2 EDU. Nothing in come-here
sends SelectMode: SelectMode('normal') wedges the switcher until a power cycle.

    ros2 run come_here_behavior check_motion_mode            # exits 0 only for mcf
    ros2 run come_here_behavior check_motion_mode --required ai
"""

import json
from typing import Optional, Tuple

MOTION_SWITCHER_REQUEST_TOPIC = '/api/motion_switcher/request'
MOTION_SWITCHER_RESPONSE_TOPIC = '/api/motion_switcher/response'
MOTION_SWITCHER_CHECK_MODE_API_ID = 1001
SPORT_API_DAMP = 1001
VALID_MOTION_MODES = ('mcf', 'ai')


def parse_mode_response(data: str) -> Optional[str]:
    """Mode name from a CheckMode response body such as {"form":"0","name":"mcf"}."""
    try:
        payload = json.loads(data)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get('name')
    return name if isinstance(name, str) else None


def mode_verdict(name: Optional[str], required: str) -> Tuple[bool, str]:
    if name is None:
        return False, 'no valid CheckMode response'
    if name != required:
        return False, f'motion mode is {name!r}, required {required!r}'
    return True, f'motion mode {name!r} verified'


def check_sport_api_ids(**api_ids) -> None:
    """Refuse a Sport API configuration that would send Damp."""
    for label, api_id in api_ids.items():
        if int(api_id) == SPORT_API_DAMP:
            raise ValueError(
                f'{label}={api_id}: api 1001 on /api/sport/request is Damp, not CheckMode'
            )


def main(argv=None) -> int:
    import argparse
    import time

    import rclpy
    from unitree_api.msg import Request, Response
    from come_here_behavior.go2_bridge_node import make_req

    parser = argparse.ArgumentParser(description='Read-only GO2 motion mode check.')
    parser.add_argument('--required', default='mcf', choices=VALID_MOTION_MODES)
    parser.add_argument('--timeout', type=float, default=6.0)
    args = parser.parse_args(argv)

    rclpy.init()
    node = rclpy.create_node('come_here_motion_mode_check')
    names = []

    def on_response(msg):
        if (msg.header.identity.api_id == MOTION_SWITCHER_CHECK_MODE_API_ID
                and msg.header.status.code == 0):
            names.append(parse_mode_response(msg.data))

    publisher = node.create_publisher(Request, MOTION_SWITCHER_REQUEST_TOPIC, 10)
    node.create_subscription(Response, MOTION_SWITCHER_RESPONSE_TOPIC, on_response, 10)
    deadline = time.monotonic() + args.timeout
    last_request = 0.0
    try:
        while time.monotonic() < deadline and not names:
            if time.monotonic() - last_request > 1.0:
                publisher.publish(make_req(MOTION_SWITCHER_CHECK_MODE_API_ID))
                last_request = time.monotonic()
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

    ok, text = mode_verdict(names[0] if names else None, args.required)
    print(f'MOTION MODE: {"PASS" if ok else "FAIL"} ({text})')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
