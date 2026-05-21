"""Unit tests for safety_limits — pure numeric motion guards (no ROS)."""

from come_here_behavior.safety_limits import clamp, clamp_velocity


def test_clamp_within_range_unchanged():
    assert clamp(0.5, -1.0, 1.0) == 0.5


def test_clamp_above_high_returns_high():
    assert clamp(5.0, -1.0, 1.0) == 1.0


def test_clamp_below_low_returns_low():
    assert clamp(-5.0, -1.0, 1.0) == -1.0


def test_clamp_velocity_within_limits_unchanged():
    vx, yaw = clamp_velocity(0.6, 1.0, max_vx=1.0, max_yaw_rate=2.5)
    assert vx == 0.6
    assert yaw == 1.0


def test_clamp_velocity_caps_excess_forward_speed():
    vx, yaw = clamp_velocity(50.0, 0.0, max_vx=1.0, max_yaw_rate=2.5)
    assert vx == 1.0
    assert yaw == 0.0


def test_clamp_velocity_caps_excess_negative_yaw():
    vx, yaw = clamp_velocity(0.0, -9.0, max_vx=1.0, max_yaw_rate=2.5)
    assert yaw == -2.5
    assert vx == 0.0


def test_clamp_velocity_nan_collapses_to_stop():
    assert clamp_velocity(float('nan'), 0.5, max_vx=1.0, max_yaw_rate=2.5) == (0.0, 0.0)


def test_clamp_velocity_inf_collapses_to_stop():
    assert clamp_velocity(0.5, float('inf'), max_vx=1.0, max_yaw_rate=2.5) == (0.0, 0.0)
