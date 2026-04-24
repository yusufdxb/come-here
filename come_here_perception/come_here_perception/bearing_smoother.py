"""EMA low-pass filter for YOLO bearing estimates.

YOLO bbox center jumps ±0.6 rad between consecutive frames even when the
target is stationary.  A simple exponential moving average at the perception
publish rate (10 Hz) suppresses this jitter while tracking real motion.
"""

from __future__ import annotations


class BearingSmoother:
    """Exponential moving average filter for bearing (radians).

    Parameters
    ----------
    alpha : float
        Smoothing factor in (0, 1].  Lower = heavier smoothing.
        At 10 Hz publish rate, α=0.3 gives τ ≈ 0.33 s.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        self._alpha = alpha
        self._value: float | None = None

    def update(self, bearing_rad: float, detected: bool) -> float:
        """Return the smoothed bearing.

        On first detection (or after a detection gap) the filter seeds from
        the raw value so there is no lag from stale state.  While the target
        is continuously detected the EMA tracks normally.  On detection loss
        the filter resets so the next appearance starts fresh.
        """
        if not detected:
            self._value = None
            return bearing_rad
        if self._value is None:
            self._value = bearing_rad
        else:
            a = self._alpha
            self._value = a * bearing_rad + (1.0 - a) * self._value
        return self._value

    def reset(self) -> None:
        self._value = None
