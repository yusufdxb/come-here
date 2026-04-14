"""MediaPipe short-range face detector.

Wraps mediapipe.solutions.face_detection for single-shot inference on a BGR
frame. CPU-only; suitable for Jetson when YOLO already owns the GPU.
"""

from typing import Optional

import numpy as np

from come_here_perception.face_detector import (
    EMPTY_RESULT,
    FaceDetectionResult,
    FaceDetector,
)


class MediapipeFaceDetector(FaceDetector):
    def __init__(self, min_detection_confidence: float = 0.5, model_selection: int = 0):
        self._min_conf = min_detection_confidence
        self._model_selection = model_selection
        self._mp_face = None
        self._detector = None

    def setup(self) -> None:
        import mediapipe as mp
        self._mp_face = mp.solutions.face_detection
        self._detector = self._mp_face.FaceDetection(
            min_detection_confidence=self._min_conf,
            model_selection=self._model_selection,
        )

    def detect(self, bgr_frame: np.ndarray) -> FaceDetectionResult:
        if self._detector is None or bgr_frame is None:
            return EMPTY_RESULT

        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        detections = results.detections or []

        if not detections:
            return EMPTY_RESULT

        best = _largest(detections)
        bbox = best.location_data.relative_bounding_box
        cx = float(bbox.xmin + bbox.width / 2.0)
        cy = float(bbox.ymin + bbox.height / 2.0)
        conf = float(best.score[0]) if best.score else 0.0

        return FaceDetectionResult(
            face_present=True,
            face_count=len(detections),
            max_confidence=conf,
            center_x_norm=_clamp01(cx),
            center_y_norm=_clamp01(cy),
        )

    def teardown(self) -> None:
        if self._detector is not None:
            self._detector.close()
        self._detector = None


def _largest(detections):
    def area(d):
        b = d.location_data.relative_bounding_box
        return b.width * b.height
    return max(detections, key=area)


def _clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v
