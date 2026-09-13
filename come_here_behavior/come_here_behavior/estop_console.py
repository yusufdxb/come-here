"""Operator e-stop console.

A long-running process that has already discovered the come-here nodes, so an
e-stop is not delayed by DDS discovery the way a one-shot `ros2 topic pub` is.
Start it in its own terminal before any live trial:

    ros2 run come_here_behavior estop_console

    Enter or e   ENGAGE the e-stop (latched in behavior_node and go2_bridge_node)
    release      release it; motion stays blocked until the next "come here"
    s            show subscriber count and the latest bridge status
    q            quit (leaves the e-stop as it is)

The physical remote is still the primary stop.
"""

import json
import threading
import time

ENGAGE = 'engage'
RELEASE = 'release'
STATUS = 'status'
QUIT = 'quit'


def parse_command(line: str):
    text = (line or '').strip().lower()
    if text in ('', 'e', 'estop', 'stop'):
        return ENGAGE
    if text == 'release':
        return RELEASE
    if text in ('s', 'status'):
        return STATUS
    if text in ('q', 'quit', 'exit'):
        return QUIT
    return None


def main(argv=None) -> int:
    import rclpy
    from std_msgs.msg import Bool, String

    rclpy.init()
    node = rclpy.create_node('come_here_estop_console')
    publisher = node.create_publisher(Bool, '/come_here/estop', 10)
    latest = {}

    def on_status(msg):
        try:
            latest.update(json.loads(msg.data))
        except ValueError:
            pass

    node.create_subscription(String, '/come_here/bridge_status', on_status, 10)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    def send(value: bool) -> None:
        msg = Bool()
        msg.data = value
        for _ in range(3):
            publisher.publish(msg)
            time.sleep(0.05)

    def subscribers() -> int:
        return node.count_subscribers('/come_here/estop')

    print(__doc__)
    time.sleep(1.5)
    print(f'subscribers on /come_here/estop: {subscribers()} '
          '(expect 2: behavior_node and go2_bridge_node)')
    try:
        while True:
            try:
                line = input('estop> ')
            except EOFError:
                break
            command = parse_command(line)
            if command == ENGAGE:
                send(True)
                print('E-STOP ENGAGED')
            elif command == RELEASE:
                send(False)
                print('e-stop released; motion needs a new "come here"')
            elif command == STATUS:
                print(f'subscribers: {subscribers()}  bridge: {latest or "no status yet"}')
            elif command == QUIT:
                break
            else:
                print('commands: Enter/e engage, release, s status, q quit')
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
