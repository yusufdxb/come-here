"""Person detection abstraction.

Provides an interface for detecting and localizing a person in the robot's
field of view. Real implementations will wrap YOLO, MediaPipe, or similar.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PersonEstimate:
    """Detected person location relative to robot."""
    bearing_rad: float    # horizontal angle to person (0 = center of FOV)
    distance_m: float     # estimated distance (0 if unknown)
    confidence: float     # 0.0 to 1.0
    detected: bool        # true if person found


class PersonDetector(ABC):
    """Interface for visual person detection."""

    @abstractmethod
    def setup(self) -> None:
        ...

    @abstractmethod
    def detect(self) -> PersonEstimate:
        """Return detection result. detected=False if no person found."""
        ...

    @abstractmethod
    def teardown(self) -> None:
        ...


class MockPersonDetector(PersonDetector):
    """Stub detector that returns a fixed person location.

    For local development without a camera.
    """

    def __init__(self, bearing_rad: float = 0.0, distance_m: float = 2.0):
        self._bearing = bearing_rad
        self._distance = distance_m
        self._detected = False

    def setup(self) -> None:
        pass

    def set_detected(self, detected: bool, bearing_rad: float = 0.0, distance_m: float = 2.0):
        """Simulate person appearing/disappearing."""
        self._detected = detected
        self._bearing = bearing_rad
        self._distance = distance_m

    def detect(self) -> PersonEstimate:
        if self._detected:
            return PersonEstimate(
                bearing_rad=self._bearing,
                distance_m=self._distance,
                confidence=0.9,
                detected=True,
            )
        return PersonEstimate(
            bearing_rad=0.0, distance_m=0.0, confidence=0.0, detected=False
        )

    def teardown(self) -> None:
        pass
