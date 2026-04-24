"""Tests for BearingSmoother EMA filter."""

import pytest

from come_here_perception.bearing_smoother import BearingSmoother


def test_seeds_on_first_detection():
    s = BearingSmoother(alpha=0.3)
    out = s.update(0.5, detected=True)
    assert out == 0.5


def test_smooths_subsequent_readings():
    s = BearingSmoother(alpha=0.3)
    s.update(0.0, detected=True)
    out = s.update(1.0, detected=True)
    assert out == pytest.approx(0.3)
    out = s.update(1.0, detected=True)
    assert out == pytest.approx(0.3 * 1.0 + 0.7 * 0.3)


def test_resets_on_detection_loss():
    s = BearingSmoother(alpha=0.3)
    s.update(0.5, detected=True)
    s.update(0.5, detected=True)
    s.update(0.0, detected=False)
    # Next detection should seed fresh, not smooth from old value
    out = s.update(-0.3, detected=True)
    assert out == -0.3


def test_alpha_one_is_passthrough():
    s = BearingSmoother(alpha=1.0)
    s.update(0.5, detected=True)
    out = s.update(-0.8, detected=True)
    assert out == -0.8


def test_alpha_near_zero_holds_initial():
    s = BearingSmoother(alpha=0.01)
    s.update(1.0, detected=True)
    out = s.update(0.0, detected=True)
    assert out == pytest.approx(0.01 * 0.0 + 0.99 * 1.0)


def test_not_detected_returns_raw():
    s = BearingSmoother(alpha=0.3)
    out = s.update(0.42, detected=False)
    assert out == 0.42


def test_jitter_suppression():
    """Simulate the hardware scenario: ±0.6 rad jitter around 0.0."""
    s = BearingSmoother(alpha=0.3)
    readings = [0.6, -0.6, 0.5, -0.5, 0.6, -0.6, 0.4, -0.4, 0.5, -0.5]
    outputs = [s.update(r, detected=True) for r in readings]
    # After 10 noisy readings centered on 0, the smoothed value should be
    # much closer to 0 than the raw ±0.6 swings
    assert abs(outputs[-1]) < 0.3


def test_reset_clears_state():
    s = BearingSmoother(alpha=0.3)
    s.update(1.0, detected=True)
    s.reset()
    out = s.update(-1.0, detected=True)
    assert out == -1.0
