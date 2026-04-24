"""ROS 2 node for visual person detection.

Publishes:
  /come_here/person_detection  (std_msgs/Float64MultiArray) [bearing, distance, confidence, detected]

Subscribes:
  /camera/image_raw            (sensor_msgs/Image) - from go2_av_node
  /utlidar/cloud_base          (sensor_msgs/PointCloud2) - base-frame LiDAR (GO2 L1)
  /come_here/mock_person       (std_msgs/Bool) - toggle mock person detection
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool, Float64MultiArray

from come_here_perception.lidar_distance_resolver import LidarDistanceResolver
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
        # YOLO input size: 320 = fast (~60ms), 640 = accurate (~200ms). Tune down
        # for bearing-tracking responsiveness in APPROACH_PERSON.
        self.declare_parameter('yolo_imgsz', 320)
        self.declare_parameter('use_lidar_distance', True)
        self.declare_parameter('lidar_cloud_topic', '/utlidar/cloud_base')
        self.declare_parameter('lidar_max_age_s', 2.0)

        use_mock = self.get_parameter('use_mock').value
        rate_hz = self.get_parameter('publish_rate_hz').value

        if use_mock:
            self._detector: PersonDetector = MockPersonDetector()
            self.get_logger().info('Using MOCK person detector (no real camera)')
        else:
            from come_here_perception.yolo_person_detector import YoloPersonDetector
            model_path = self.get_parameter('model_path').value
            confidence = self.get_parameter('confidence').value
            imgsz = int(self.get_parameter('yolo_imgsz').value)
            self._detector: PersonDetector = YoloPersonDetector(
                model_path=model_path, confidence=confidence, imgsz=imgsz,
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

        self._use_lidar_distance = bool(self.get_parameter('use_lidar_distance').value)
        self._lidar_max_age_s = float(self.get_parameter('lidar_max_age_s').value)
        self.declare_parameter('lidar_min_vertical_extent_m', 0.15)
        self.declare_parameter('lidar_min_points', 4)
        self.declare_parameter('lidar_z_min', 0.1)
        # Cone half-angle must be wide enough to cover the operator's leg
        # stance at close range. At 0.8 m a ~0.3 m stance subtends ±0.19 rad;
        # the previous 0.14 rad half-cone dropped leg points at the edges and
        # caused repeated gate failures → bbox fallback → stop-distance miss.
        self.declare_parameter('lidar_cone_half_rad', 0.30)
        min_extent = float(self.get_parameter('lidar_min_vertical_extent_m').value)
        min_pts = int(self.get_parameter('lidar_min_points').value)
        z_min = float(self.get_parameter('lidar_z_min').value)
        self._cone_half_rad = float(self.get_parameter('lidar_cone_half_rad').value)
        self._resolver = LidarDistanceResolver(
            cone_half_rad=self._cone_half_rad,
            min_points=min_pts,
            min_vertical_extent_m=min_extent,
            z_min=z_min,
        )
        self._latest_cloud_xyz: np.ndarray | None = None
        self._latest_cloud_stamp_s: float = 0.0
        self._lidar_fallback_logged: bool = False
        # Debug counters (print once per second)
        self._cloud_cb_count: int = 0
        self._tick_count: int = 0
        self._lidar_success_count: int = 0
        self._bbox_fallback_count: int = 0

        if self._use_lidar_distance and not use_mock:
            lidar_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            )
            cloud_topic = self.get_parameter('lidar_cloud_topic').value
            self.create_subscription(
                PointCloud2, cloud_topic, self._on_cloud_base, lidar_qos
            )
            self.get_logger().info(f'Subscribed to {cloud_topic} for distance refinement')
        elif use_mock:
            self.get_logger().info('Mock mode — skipping LiDAR distance refinement')
        else:
            self.get_logger().info('LiDAR distance refinement disabled (use_lidar_distance=false)')

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

    def _on_cloud_base(self, msg: PointCloud2):
        """Parse the cloud once on arrival; cache XYZ as a contiguous (N,3) array.

        Zero-copy parse: point_step=32 on the GO2 L1 cloud_base means each
        point is 8 x float32 slots; the first three slots are x, y, z.
        We read all 8 columns and view the first three.
        """
        if msg.point_step != 32:
            self.get_logger().warn(
                f'cloud_base point_step={msg.point_step}, expected 32 — skipping'
            )
            return
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 8)
        self._latest_cloud_xyz = arr[:, :3].copy()
        # The Unitree bare-DDS lidar stamps with its own unsynchronised clock
        # (~188 days behind Jetson system time in practice). Use arrival time
        # on the Jetson side instead — simpler and robust to that skew.
        self._latest_cloud_stamp_s = self.get_clock().now().nanoseconds * 1e-9
        self._cloud_cb_count += 1

    def _mock_person_cb(self, msg: Bool):
        if isinstance(self._detector, MockPersonDetector):
            self._detector.set_detected(msg.data)
            self.get_logger().info(f'Mock person detected: {msg.data}')

    def _tick(self):
        result = self._detector.detect()
        distance_m = result.distance_m

        if result.detected and self._use_lidar_distance:
            now_s = self.get_clock().now().nanoseconds * 1e-9
            cloud_age_s = now_s - self._latest_cloud_stamp_s
            cloud_fresh = (
                self._latest_cloud_xyz is not None
                and cloud_age_s < self._lidar_max_age_s
            )
            if cloud_fresh:
                refined = self._resolver.refine(
                    bearing_rad=result.bearing_rad,
                    cloud_xyz=self._latest_cloud_xyz,
                )
                if refined is not None:
                    distance_m = refined
                    self._lidar_success_count += 1
                    if self._lidar_fallback_logged:
                        self.get_logger().info(
                            f'lidar distance recovered: {refined:.2f} m'
                        )
                        self._lidar_fallback_logged = False
                else:
                    self._bbox_fallback_count += 1
                    if not self._lidar_fallback_logged:
                        self.get_logger().info(
                            f'lidar gate failed (bearing={result.bearing_rad:.2f}) — '
                            f'falling back to bbox distance {result.distance_m:.2f} m'
                        )
                        self._lidar_fallback_logged = True
            else:
                self._bbox_fallback_count += 1
                if not self._lidar_fallback_logged:
                    self.get_logger().info(
                        f'no fresh cloud (age={cloud_age_s:.2f}s) — bbox distance '
                        f'{result.distance_m:.2f} m'
                    )
                    self._lidar_fallback_logged = True

        self._tick_count += 1
        # Every 10 ticks, log counters + diagnostic measurements
        if self._tick_count % 10 == 0:
            now_s = self.get_clock().now().nanoseconds * 1e-9
            age = now_s - self._latest_cloud_stamp_s if self._latest_cloud_stamp_s > 0 else -1
            n_pts = int(self._latest_cloud_xyz.shape[0]) if self._latest_cloud_xyz is not None else 0
            # Probe wedge without gates (use zero gates) to see raw count in wedge
            wedge_count = 0
            z_extent = 0.0
            wedge_r = 0.0
            if self._latest_cloud_xyz is not None:
                x = self._latest_cloud_xyz[:, 0]
                y = self._latest_cloud_xyz[:, 1]
                z = self._latest_cloud_xyz[:, 2]
                az = np.arctan2(y, x)
                m = (np.abs(az - result.bearing_rad) < self._cone_half_rad) & (z > 0.1) & (z < 1.8) & (x > 0.2)
                wedge_count = int(m.sum())
                if wedge_count > 0:
                    zw = z[m]
                    z_extent = float(zw.max() - zw.min())
                    rw = np.hypot(x[m], y[m])
                    wedge_r = float(np.percentile(rw, 10.0))
            self.get_logger().info(
                f'[dbg] t={self._tick_count} cb={self._cloud_cb_count} '
                f'hits={self._lidar_success_count} fb={self._bbox_fallback_count} '
                f'age={age:.2f}s pts={n_pts} wedge@{result.bearing_rad:.2f}={wedge_count} '
                f'zext={z_extent:.2f} wR={wedge_r:.2f} '
                f'dist={distance_m:.2f} det={int(result.detected)}'
            )

        msg = Float64MultiArray()
        msg.data = [
            result.bearing_rad,
            distance_m,
            result.confidence,
            float(result.detected),
            float(result.bbox_h_frac),
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
