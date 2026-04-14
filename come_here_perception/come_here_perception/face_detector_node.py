"""ROS 2 node for on-demand face detection.

Behavior:
  - Subscribes to /camera/image_raw and buffers the latest frame only.
  - Stays idle (zero inference) until a True message arrives on
    /come_here/face_detect_request.
  - On trigger: runs one inference on the latest frame, publishes a
    FaceDetection result on /come_here/face_detection, and returns to idle.

This keeps CPU free for YOLO + Whisper during the approach walk; the face
detector only runs once per demo cycle, after the dog sits.

Subscribes:
  /camera/image_raw                (sensor_msgs/Image)
  /come_here/face_detect_request   (std_msgs/Bool)

Publishes:
  /come_here/face_detection        (come_here_msgs/FaceDetection)
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from come_here_msgs.msg import FaceDetection
from come_here_perception.face_detector import (
    EMPTY_RESULT,
    FaceDetector,
    MockFaceDetector,
)


class FaceDetectorNode(Node):
    def __init__(self):
        super().__init__('face_detector_node')

        self.declare_parameter('face_detector', 'mediapipe')  # 'mediapipe' | 'mock'
        self.declare_parameter('min_confidence', 0.5)

        impl = self.get_parameter('face_detector').value
        min_conf = self.get_parameter('min_confidence').value

        self._detector: FaceDetector = self._build_detector(impl, min_conf)
        self._detector.setup()
        self.get_logger().info(f'Face detector: {impl} (min_conf={min_conf})')

        cam_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._last_frame = None
        self.create_subscription(Image, '/camera/image_raw', self._on_image, cam_qos)
        self.create_subscription(
            Bool, '/come_here/face_detect_request', self._on_request, 10
        )
        self._pub = self.create_publisher(
            FaceDetection, '/come_here/face_detection', 10
        )

    def _build_detector(self, impl: str, min_conf: float) -> FaceDetector:
        if impl == 'mock':
            return MockFaceDetector()
        if impl == 'mediapipe':
            from come_here_perception.mediapipe_face_detector import (
                MediapipeFaceDetector,
            )
            return MediapipeFaceDetector(min_detection_confidence=min_conf)
        self.get_logger().warn(f'Unknown face_detector "{impl}", using mock')
        return MockFaceDetector()

    def _on_image(self, msg: Image):
        try:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )
            self._last_frame = frame
        except (ValueError, TypeError) as e:
            self.get_logger().warn(f'Bad image frame: {e}')

    def _on_request(self, msg: Bool):
        if not msg.data:
            return

        if self._last_frame is None:
            self.get_logger().warn('Face detect requested but no frame buffered')
            self._publish(EMPTY_RESULT)
            return

        result = self._detector.detect(self._last_frame)
        self.get_logger().info(
            f'Face: present={result.face_present} count={result.face_count} '
            f'conf={result.max_confidence:.2f} center=({result.center_x_norm:.2f},'
            f'{result.center_y_norm:.2f})'
        )
        self._publish(result)

    def _publish(self, result):
        msg = FaceDetection()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.face_present = bool(result.face_present)
        msg.face_count = int(result.face_count)
        msg.max_confidence = float(result.max_confidence)
        msg.center_x_norm = float(result.center_x_norm)
        msg.center_y_norm = float(result.center_y_norm)
        self._pub.publish(msg)

    def destroy_node(self):
        self._detector.teardown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
