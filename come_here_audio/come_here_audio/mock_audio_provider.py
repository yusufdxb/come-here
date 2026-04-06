"""Mock audio direction provider for local development and testing.

Publishes a fixed or configurable direction estimate without real hardware.
"""

import math
import random

from come_here_audio.audio_direction_provider import AudioDirectionProvider, DirectionEstimate


class MockAudioProvider(AudioDirectionProvider):
    """Simulates a microphone array by returning synthetic direction estimates."""

    def __init__(self, fixed_azimuth_rad: float = 0.0, noise_std: float = 0.05):
        self._fixed_azimuth = fixed_azimuth_rad
        self._noise_std = noise_std
        self._active = False

    def setup(self) -> None:
        self._active = True

    def get_direction(self) -> DirectionEstimate | None:
        if not self._active:
            return None
        noise = random.gauss(0, self._noise_std)
        return DirectionEstimate(
            azimuth_rad=self._fixed_azimuth + noise,
            confidence=max(0.0, min(1.0, 0.85 + random.gauss(0, 0.05))),
        )

    def teardown(self) -> None:
        self._active = False
