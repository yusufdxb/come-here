"""ROS 2 node for visual person detection.

Publishes:
  /come_here/person_detection  (std_msgs/Float64MultiArray) [bearing, distance, confidence, detected]

Subscribes:
  /come_here/mock_person       (std_msgs/Bool) - toggle mock person detection
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray

from come_here_perception.person_detector import MockPersonDetector, PersonDetector


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.declare_parameter('use_mock', True)
        self.declare_parameter('publish_rate_hz', 10.0)

        use_mock = self.get_parameter('use_mock').value
        rate_hz = self.get_parameter('publish_rate_hz').value

        # TODO: Add real detector (YOLO, MediaPipe, etc.) when camera is available
        if use_mock:
            self._detector: PersonDetector = MockPersonDetector()
            self.get_logger().info('Using MOCK person detector (no real camera)')
        else:
            raise NotImplementedError(
                'Real person detector not yet implemented. Set use_mock:=true.'
            )

        self._detector.setup()

        self._pub = self.create_publisher(
            Float64MultiArray, '/come_here/person_detection', 10
        )

        if use_mock:
            self._mock_sub = self.create_subscription(
                Bool, '/come_here/mock_person', self._mock_person_cb, 10
            )

        self._timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(f'Perception node started at {rate_hz} Hz')

    def _mock_person_cb(self, msg: Bool):
        if isinstance(self._detector, MockPersonDetector):
            self._detector.set_detected(msg.data)
            self.get_logger().info(f'Mock person detected: {msg.data}')

    def _tick(self):
        result = self._detector.detect()
        msg = Float64MultiArray()
        msg.data = [
            result.bearing_rad,
            result.distance_m,
            result.confidence,
            float(result.detected),
        ]
        self._pub.publish(msg)

    def destroy_node(self):
        self._detector.teardown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
