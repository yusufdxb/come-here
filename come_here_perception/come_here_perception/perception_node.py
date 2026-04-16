"""ROS 2 node for visual person detection.

Publishes:
  /come_here/person_detection  (std_msgs/Float64MultiArray) [bearing, distance, confidence, detected]

Subscribes:
  /camera/image_raw            (sensor_msgs/Image) - from go2_av_node
  /come_here/mock_person       (std_msgs/Bool) - toggle mock person detection
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64MultiArray

from come_here_perception.person_detector import MockPersonDetector, PersonDetector

# YoloPersonDetector pulls in ultralytics (and OpenCV). It is imported lazily
# inside the non-mock branch so mock-mode launches do not require those
# packages to be installed.


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.declare_parameter('use_mock', False)
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('model_path', '/home/unitree/come-here/models/yolo11n.pt')
        self.declare_parameter('confidence', 0.45)

        use_mock = self.get_parameter('use_mock').value
        rate_hz = self.get_parameter('publish_rate_hz').value

        if use_mock:
            self._detector: PersonDetector = MockPersonDetector()
            self.get_logger().info('Using MOCK person detector (no real camera)')
        else:
            from come_here_perception.yolo_person_detector import YoloPersonDetector
            model_path = self.get_parameter('model_path').value
            confidence = self.get_parameter('confidence').value
            self._detector: PersonDetector = YoloPersonDetector(
                model_path=model_path, confidence=confidence,
            )
            self.get_logger().info(f'Using YOLO person detector: {model_path}')

            # Subscribe to camera feed
            cam_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.create_subscription(
                Image, '/camera/image_raw', self._on_image, cam_qos
            )
            self.get_logger().info('Subscribed to /camera/image_raw')

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

    def _on_image(self, msg: Image):
        """Convert ROS Image to numpy and feed to detector."""
        # Only real detectors expose update_frame; mock mode never subscribes.
        if not hasattr(self._detector, 'update_frame'):
            return
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )
        self._detector.update_frame(frame)

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
        # A second SIGINT can land during teardown or during interpreter
        # shutdown (e.g. threading._shutdown). Ignore it for the rest of the
        # process so shutdown stays quiet.
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
