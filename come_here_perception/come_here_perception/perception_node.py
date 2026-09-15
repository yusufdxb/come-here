"""ROS 2 node for visual person detection.

Publishes:
  /come_here/person_detection  (std_msgs/Float64MultiArray)
      [bearing_rad, distance_m, confidence, detected, bbox_h_frac,
       distance_source, frame_age_s]
      distance_source: 0 none, 1 bbox pinhole, 2 LiDAR
      The person is the one candidate_selector picks inside the behavior
      node's gate; with no fresh gate, the tallest box (camera-only demo).
  /come_here/person_boxes      (std_msgs/Float64MultiArray) for the operator view:
      [img_w, img_h, n, selected_index (-1 none), gate_center_rad, gate_half_rad
       (nan = no gate), then per box x1, y1, x2, y2, confidence, bearing_rad, in_gate]

Subscribes:
  /camera/image_raw            (sensor_msgs/Image) - GO2 front camera publisher
  /utlidar/cloud_base          (sensor_msgs/PointCloud2) - base-frame LiDAR (GO2 L1)
  /come_here/mock_person       (std_msgs/Bool) - toggle mock person detection
  /come_here/target_gate       (std_msgs/Float64MultiArray) [center_rad, half_width_rad]

Freshness rules (fail closed):
  * YOLO runs once per NEW camera frame. Re-running it on the same frame would
    let one image count as several consecutive detections.
  * If no frame has arrived within max_frame_age_s the node publishes an
    explicit not-detected message, so a dead camera stops the robot instead
    of freezing the last detection in place.

Bearing is published raw: the behavior node applies the single EMA layer.
"""

import math
import time

import numpy as np

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool, Float64MultiArray

from come_here_perception.bearing_smoother import BearingSmoother
from come_here_perception.candidate_selector import largest, select
from come_here_perception.lidar_distance_resolver import LidarDistanceResolver
from come_here_perception.person_detector import (
    MockPersonDetector, PersonDetector, PersonEstimate,
)

# YoloPersonDetector pulls in ultralytics (and OpenCV). It is imported lazily
# inside the non-mock branch so mock-mode launches do not require those
# packages to be installed.

DISTANCE_SOURCE_NONE = 0.0
DISTANCE_SOURCE_BBOX = 1.0
DISTANCE_SOURCE_LIDAR = 2.0


class PerceptionNode(Node):
    def __init__(self, detector: PersonDetector = None, **kwargs):
        super().__init__('perception_node', **kwargs)

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
        # A camera frame older than this is treated as no frame at all. Short:
        # until it expires nothing is published, so the robot walks on.
        self.declare_parameter('max_frame_age_s', 0.5)
        # 1.0 = pass-through. The behavior node owns bearing smoothing; a
        # second EMA here (re-added by an April merge, never run on hardware)
        # doubles the lag the ALIGN deadband was tuned against.
        self.declare_parameter('bearing_ema_alpha', 1.0)
        # A gate older than this is ignored (behavior_node republishes it every tick).
        self.declare_parameter('gate_max_age_s', 0.5)

        use_mock = self.get_parameter('use_mock').value
        rate_hz = self.get_parameter('publish_rate_hz').value
        self._max_frame_age_s = float(self.get_parameter('max_frame_age_s').value)
        self._now = time.monotonic

        if detector is not None:
            self._detector: PersonDetector = detector
            self._uses_camera = True
        elif use_mock:
            self._detector = MockPersonDetector()
            self._uses_camera = False
            self.get_logger().info('Using MOCK person detector (no real camera)')
        else:
            from come_here_perception.yolo_person_detector import YoloPersonDetector
            model_path = self.get_parameter('model_path').value
            confidence = self.get_parameter('confidence').value
            imgsz = int(self.get_parameter('yolo_imgsz').value)
            self._detector = YoloPersonDetector(
                model_path=model_path, confidence=confidence, imgsz=imgsz,
            )
            self._uses_camera = True
            self.get_logger().info(f'Using YOLO person detector: {model_path}')

        self._frame_seq = 0
        self._last_processed_seq = 0
        self._frame_rx_s = None
        self._last_warn_s = {}
        if self._uses_camera:
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
        bearing_alpha = float(self.get_parameter('bearing_ema_alpha').value)
        self._bearing_smoother = BearingSmoother(alpha=bearing_alpha)
        if bearing_alpha < 1.0:
            self.get_logger().warn(
                f'perception bearing_ema_alpha={bearing_alpha}: bearing is smoothed '
                'here AND in behavior_node (double smoothing)'
            )

        self.declare_parameter('lidar_min_vertical_extent_m', 0.15)
        self.declare_parameter('lidar_min_points', 4)
        self.declare_parameter('lidar_z_min', 0.1)
        # Cone half-angle must be wide enough to cover the operator's leg
        # stance at close range. At 0.8 m a ~0.3 m stance subtends ±0.19 rad;
        # the previous 0.14 rad half-cone dropped leg points at the edges and
        # caused repeated gate failures, then bbox fallback, then stop-distance miss.
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

        if self._use_lidar_distance and self._uses_camera:
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
            self.get_logger().info('Mock mode, skipping LiDAR distance refinement')
        else:
            self.get_logger().info('LiDAR distance refinement disabled (use_lidar_distance=false)')

        self._pub = self.create_publisher(
            Float64MultiArray, '/come_here/person_detection', 10
        )
        self._boxes_pub = self.create_publisher(
            Float64MultiArray, '/come_here/person_boxes', 10
        )
        self._gate_max_age_s = float(self.get_parameter('gate_max_age_s').value)
        self._gate = None
        self._gate_rx_s = None
        self.create_subscription(
            Float64MultiArray, '/come_here/target_gate', self._on_gate, 10
        )

        if use_mock and detector is None:
            self._mock_sub = self.create_subscription(
                Bool, '/come_here/mock_person', self._mock_person_cb, 10
            )

        self._timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f'Perception node started at {rate_hz} Hz '
            f'(max_frame_age_s={self._max_frame_age_s})'
        )

    def _warn_throttled(self, key: str, text: str, period_s: float = 2.0) -> None:
        now = self._now()
        if now - self._last_warn_s.get(key, -math.inf) >= period_s:
            self._last_warn_s[key] = now
            self.get_logger().warn(text)

    def _info_throttled(self, key: str, text: str, period_s: float = 1.0) -> None:
        now = self._now()
        if now - self._last_warn_s.get(key, -math.inf) >= period_s:
            self._last_warn_s[key] = now
            self.get_logger().info(text)

    def _on_image(self, msg: Image):
        """Convert ROS Image to numpy and feed to detector."""
        # Only real detectors expose update_frame; mock mode never subscribes.
        if not hasattr(self._detector, 'update_frame'):
            return
        try:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )
        except ValueError as exc:
            self._warn_throttled('bad_frame', f'Dropping malformed camera frame: {exc}')
            return
        self._detector.update_frame(frame)
        self._frame_seq += 1
        self._frame_rx_s = self._now()

    def _on_cloud_base(self, msg: PointCloud2):
        """Parse the cloud once on arrival; cache XYZ as a contiguous (N,3) array.

        Zero-copy parse: point_step=32 on the GO2 L1 cloud_base means each
        point is 8 x float32 slots; the first three slots are x, y, z.
        We read all 8 columns and view the first three.
        """
        if msg.point_step != 32:
            self.get_logger().warn(
                f'cloud_base point_step={msg.point_step}, expected 32, skipping'
            )
            return
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 8)
        self._latest_cloud_xyz = arr[:, :3].copy()
        # The Unitree bare-DDS lidar stamps with its own unsynchronised clock
        # (~188 days behind Jetson system time in practice). Use arrival time
        # on the Jetson side instead, simpler and robust to that skew.
        self._latest_cloud_stamp_s = self._now()
        self._cloud_cb_count += 1

    def _on_gate(self, msg: Float64MultiArray):
        data = list(msg.data)
        if len(data) != 2 or not all(math.isfinite(v) for v in data) or data[1] < 0.0:
            self._warn_throttled('bad_gate', f'Ignoring malformed target_gate {data}')
            return
        self._gate = (float(data[0]), float(data[1]))
        self._gate_rx_s = self._now()

    def _fresh_gate(self):
        if self._gate_rx_s is None or self._now() - self._gate_rx_s > self._gate_max_age_s:
            return None
        return self._gate

    def _publish_boxes(self, candidates, selected, in_gate, gate):
        size = self._detector.frame_size() or (0, 0)
        nan = float('nan')
        data = [float(size[0]), float(size[1]), float(len(candidates)),
                float(-1 if selected is None else selected),
                nan if gate is None else gate[0], nan if gate is None else gate[1]]
        for c, ok in zip(candidates, in_gate):
            data.extend([*(float(v) for v in c.box_px), float(c.confidence),
                         float(c.bearing_rad), float(ok)])
        msg = Float64MultiArray()
        msg.data = data
        self._boxes_pub.publish(msg)

    def _mock_person_cb(self, msg: Bool):
        if isinstance(self._detector, MockPersonDetector):
            self._detector.set_detected(msg.data)
            self.get_logger().info(f'Mock person detected: {msg.data}')

    def _publish(self, bearing, distance, confidence, detected, bbox_h_frac,
                 distance_source, frame_age_s):
        msg = Float64MultiArray()
        msg.data = [
            float(bearing), float(distance), float(confidence), float(detected),
            float(bbox_h_frac), float(distance_source), float(frame_age_s),
        ]
        self._pub.publish(msg)

    def _tick(self):
        frame_age_s = 0.0
        if self._uses_camera:
            now = self._now()
            if self._frame_rx_s is None or now - self._frame_rx_s > self._max_frame_age_s:
                age = 999.0 if self._frame_rx_s is None else now - self._frame_rx_s
                self._warn_throttled(
                    'stale_camera',
                    f'No fresh camera frame (age={age:.1f}s): publishing not-detected',
                )
                self._bearing_smoother.reset()
                self._publish(0.0, 0.0, 0.0, False, 0.0, DISTANCE_SOURCE_NONE, age)
                self._publish_boxes([], None, [], self._fresh_gate())
                return
            if self._frame_seq == self._last_processed_seq:
                return  # nothing new to look at
            self._last_processed_seq = self._frame_seq
            frame_age_s = now - self._frame_rx_s

        candidates = self._detector.detect_all()
        gate = self._fresh_gate()
        if gate is None:
            selected = largest(candidates)
            in_gate = [True] * len(candidates)
        else:
            selected, in_gate = select(candidates, gate[0], gate[1])
        if selected is None:
            result = PersonEstimate(0.0, 0.0, 0.0, False)
        else:
            c = candidates[selected]
            result = PersonEstimate(c.bearing_rad, c.distance_m, c.confidence, True, c.bbox_h_frac)
        self._publish_boxes(candidates, selected, in_gate, gate)
        if gate is not None and candidates:
            rejected = [f'{math.degrees(c.bearing_rad):+.0f}deg/{c.confidence:.2f}'
                        for c, ok in zip(candidates, in_gate) if not ok]
            chosen = ('none' if selected is None else
                      f'{math.degrees(result.bearing_rad):+.0f}deg/{result.confidence:.2f}')
            self._info_throttled(
                'gate',
                f'gate {math.degrees(gate[0]):+.0f}+/-{math.degrees(gate[1]):.0f}deg: '
                f'{len(candidates)} people, selected {chosen}, outside gate {rejected}')
        distance_m = result.distance_m
        distance_source = (
            DISTANCE_SOURCE_BBOX if result.detected and distance_m > 0.0
            else DISTANCE_SOURCE_NONE
        )

        if result.detected and self._use_lidar_distance:
            now_s = self._now()
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
                    distance_source = DISTANCE_SOURCE_LIDAR
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
                            f'lidar gate failed (bearing={result.bearing_rad:.2f}), '
                            f'falling back to bbox distance {result.distance_m:.2f} m'
                        )
                        self._lidar_fallback_logged = True
            else:
                self._bbox_fallback_count += 1
                if not self._lidar_fallback_logged:
                    self.get_logger().info(
                        f'no fresh cloud (age={cloud_age_s:.2f}s), bbox distance '
                        f'{result.distance_m:.2f} m'
                    )
                    self._lidar_fallback_logged = True

        bearing_out = self._bearing_smoother.update(
            result.bearing_rad, result.detected,
        )

        self._tick_count += 1
        # Every 10 detections, log counters + diagnostic measurements
        if self._tick_count % 10 == 0:
            now_s = self._now()
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
                f'dist={distance_m:.2f} det={int(result.detected)} '
                f'raw_b={result.bearing_rad:.2f} frame_age={frame_age_s:.2f}s'
            )

        self._publish(
            bearing_out, distance_m, result.confidence, result.detected,
            result.bbox_h_frac, distance_source, frame_age_s,
        )

    def destroy_node(self):
        self._detector.teardown()
        super().destroy_node()


def main(args=None):
    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
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
