"""Abstract interface for audio direction estimation.

Any microphone hardware integration must implement AudioDirectionProvider.
This keeps the rest of the system decoupled from specific mic hardware.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DirectionEstimate:
    """Result from a direction-of-arrival estimation."""
    azimuth_rad: float   # horizontal angle in radians (0 = forward, positive = left)
    confidence: float    # 0.0 to 1.0


class AudioDirectionProvider(ABC):
    """Interface for sound source direction estimation.

    Implementations should wrap a specific microphone array and its
    direction-of-arrival algorithm (e.g., GCC-PHAT, MUSIC, SRP-PHAT).

    To add a new microphone:
    1. Subclass AudioDirectionProvider
    2. Implement setup(), get_direction(), and teardown()
    3. Register the new provider in audio_node.py
    """

    @abstractmethod
    def setup(self) -> None:
        """Initialize hardware and algorithm resources."""
        ...

    @abstractmethod
    def get_direction(self) -> DirectionEstimate | None:
        """Return the current estimated direction of the dominant sound source.

        Returns None if no sound source is detected or confidence is too low.
        """
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Release hardware and algorithm resources."""
        ...
