"""Process lifecycle for motion-critical nodes.

By default ``rclpy.init()`` installs SIGINT and SIGTERM handlers that shut the
ROS context down before control returns to ``main()``. Any stop command that
``destroy_node()`` publishes after that point is silently dropped, so Ctrl+C
(or the SIGINT / SIGTERM that ``ros2 launch`` sends on shutdown) left the GO2
holding its last gait setpoint. Measured 2026-09-13 with an independent
subscriber: 0 of 4 signal shutdowns delivered a stop.

``run_node()`` disables the rclpy handlers and turns SIGINT / SIGTERM into
``KeyboardInterrupt``, so ``destroy_node()`` runs while the context is still
valid and its final stop reaches DDS before rclpy shuts down.
"""

import signal

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.signals import SignalHandlerOptions


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def run_node(node_factory, args=None) -> None:
    """Create a node with ``node_factory()``, spin it, and tear it down safely."""
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    node = None
    try:
        node = node_factory()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # A repeated signal must not interrupt the final stop publish.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
