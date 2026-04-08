"""YOLO-based person detector for the GO2 front camera.

Subscribes to /camera/image_raw (sensor_msgs/Image) published by go2_av_node,
runs YOLO11n inference filtering for COCO class 0 (person), and returns
bearing + estimated distance via the PersonDetector interface.

Bearing is computed from the bounding box center relative to the image center.
Distance is estimated from the bounding box height using a simple pinhole model.
"""

import math

import cv2
import numpy as np
from ultralytics import YOLO

from come_here_perception.person_detector import PersonDetector, PersonEstimate

# GO2 front camera approximate parameters
# FOV measured from videohub JPEG output (~120 deg diagonal, ~90 deg horizontal)
CAMERA_HFOV_DEG = 90.0
# Average person height in meters (used for distance estimation)
PERSON_HEIGHT_M = 1.7


class YoloPersonDetector(PersonDetector):
    """Detect persons using YOLO and estimate bearing/distance.

    Args:
        model_path: Path to YOLO .pt model file.
        confidence: Minimum detection confidence (0.0-1.0).
        camera_hfov_deg: Horizontal field of view in degrees.
    """

    def __init__(
        self,
        model_path: str = "/home/unitree/come-here/models/yolo11n.pt",
        confidence: float = 0.45,
        camera_hfov_deg: float = CAMERA_HFOV_DEG,
    ):
        self._model_path = model_path
        self._confidence = confidence
        self._hfov_rad = math.radians(camera_hfov_deg)
        self._model = None
        self._last_frame = None

    def setup(self) -> None:
        self._model = YOLO(self._model_path)

    def update_frame(self, frame: np.ndarray) -> None:
        """Feed a new BGR frame from the camera."""
        self._last_frame = frame

    def detect(self) -> PersonEstimate:
        """Run YOLO on the last frame and return the closest person."""
        if self._model is None or self._last_frame is None:
            return PersonEstimate(
                bearing_rad=0.0, distance_m=0.0, confidence=0.0, detected=False
            )

        frame = self._last_frame
        h, w = frame.shape[:2]

        results = self._model(
            frame, conf=self._confidence, verbose=False, classes=[0]
        )[0]

        if len(results.boxes) == 0:
            return PersonEstimate(
                bearing_rad=0.0, distance_m=0.0, confidence=0.0, detected=False
            )

        # Pick the largest bounding box (likely closest person)
        best_box = None
        best_area = 0
        best_conf = 0.0
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)
                best_conf = float(box.conf[0])

        x1, y1, x2, y2 = best_box
        bbox_cx = (x1 + x2) / 2.0
        bbox_h = y2 - y1

        # Bearing: pixel offset from center → angle
        # Positive = left of center (ROS convention)
        pixel_offset = (w / 2.0) - bbox_cx
        focal_length_px = (w / 2.0) / math.tan(self._hfov_rad / 2.0)
        bearing_rad = math.atan2(pixel_offset, focal_length_px)

        # Distance: pinhole model from bbox height
        # d = (real_height * focal_length) / bbox_height_px
        if bbox_h > 10:
            distance_m = (PERSON_HEIGHT_M * focal_length_px) / bbox_h
        else:
            distance_m = 0.0

        return PersonEstimate(
            bearing_rad=bearing_rad,
            distance_m=distance_m,
            confidence=best_conf,
            detected=True,
        )

    def teardown(self) -> None:
        self._model = None
        self._last_frame = None
