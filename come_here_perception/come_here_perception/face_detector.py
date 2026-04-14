"""Face detection abstraction.

Interface for single-shot face detection on a camera frame. Used by the
behavior node after the dog sits, to log whether a face is visible.

Implementations live in sibling modules:
  - MockFaceDetector: deterministic, for unit tests and dry-runs
  - MediapipeFaceDetector: MediaPipe short-range face detector (CPU)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FaceDetectionResult:
    face_present: bool
    face_count: int
    max_confidence: float
    center_x_norm: float  # normalized [0, 1] x of largest face centroid
    center_y_norm: float  # normalized [0, 1] y of largest face centroid


EMPTY_RESULT = FaceDetectionResult(
    face_present=False,
    face_count=0,
    max_confidence=0.0,
    center_x_norm=0.5,
    center_y_norm=0.5,
)


class FaceDetector(ABC):
    """Single-shot face detector. `detect(frame)` runs one inference."""

    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    def detect(self, bgr_frame: np.ndarray) -> FaceDetectionResult: ...

    @abstractmethod
    def teardown(self) -> None: ...


class MockFaceDetector(FaceDetector):
    """Returns a preconfigured result. Useful for unit tests and dry-runs."""

    def __init__(self, result: Optional[FaceDetectionResult] = None):
        self._result = result if result is not None else EMPTY_RESULT

    def set_result(self, result: FaceDetectionResult) -> None:
        self._result = result

    def setup(self) -> None:
        pass

    def detect(self, bgr_frame: np.ndarray) -> FaceDetectionResult:
        return self._result

    def teardown(self) -> None:
        pass
