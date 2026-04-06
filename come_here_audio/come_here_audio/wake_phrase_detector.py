"""Wake phrase detection abstraction.

Detects the trigger phrase "come here" from audio input.
Currently a stub - real implementation will depend on the chosen
ASR/wake-word engine (e.g., Vosk, Porcupine, NeMo).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PhraseDetection:
    """Result of wake phrase detection."""
    phrase: str
    confidence: float  # 0.0 to 1.0


class WakePhraseDetector(ABC):
    """Interface for wake phrase / command detection."""

    @abstractmethod
    def setup(self) -> None:
        ...

    @abstractmethod
    def check(self) -> PhraseDetection | None:
        """Check if the wake phrase was detected.

        Returns PhraseDetection if detected, None otherwise.
        Non-blocking - returns immediately.
        """
        ...

    @abstractmethod
    def teardown(self) -> None:
        ...


class MockWakePhraseDetector(WakePhraseDetector):
    """Stub detector that never triggers unless manually signaled.

    For testing, publish to /come_here/mock_trigger (std_msgs/Bool)
    to simulate a wake phrase detection.
    """

    def __init__(self):
        self._triggered = False

    def setup(self) -> None:
        pass

    def set_triggered(self, triggered: bool = True) -> None:
        """Called externally (e.g., from a ROS callback) to simulate detection."""
        self._triggered = triggered

    def check(self) -> PhraseDetection | None:
        if self._triggered:
            self._triggered = False
            return PhraseDetection(phrase="come here", confidence=0.95)
        return None

    def teardown(self) -> None:
        pass
