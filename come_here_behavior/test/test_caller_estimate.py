"""Odometry-frame caller estimate: explicit transform math, TTL, uncertainty, identity."""

import math

import pytest

from come_here_behavior.caller_estimate import (
    CallerEstimate,
    Pose2D,
    caller_position,
    relative_polar,
)


def test_transform_round_trip_at_many_poses():
    for yaw in (-3.0, -1.2, 0.0, 0.7, 2.9):
        for bearing in (-0.8, 0.0, 0.5):
            pose = Pose2D(1.3, -0.4, yaw)
            cx, cy = caller_position(pose, bearing, 2.5)
            b, r = relative_polar(pose, cx, cy)
            assert r == pytest.approx(2.5)
            assert math.atan2(math.sin(b - bearing), math.cos(b - bearing)) == pytest.approx(0.0,
                                                                                               abs=1e-9)


def test_positive_bearing_is_to_the_left():
    cx, cy = caller_position(Pose2D(0.0, 0.0, 0.0), math.radians(90.0), 1.0)
    assert cx == pytest.approx(0.0, abs=1e-9) and cy == pytest.approx(1.0)


def test_prediction_after_the_robot_moves_sideways():
    est = CallerEstimate()
    est.observe(Pose2D(0.0, 0.0, 0.0), 0.0, 3.0, 0.0)          # caller 3 m straight ahead
    pred = est.predict(Pose2D(1.0, 1.0, 0.0), 0.5)             # robot stepped forward-left
    assert pred.range_m == pytest.approx(math.hypot(2.0, 1.0))
    assert pred.bearing_rad == pytest.approx(math.atan2(-1.0, 2.0))   # caller now to the right


def test_prediction_expires_and_uncertainty_grows():
    est = CallerEstimate(ttl_s=1.5, base_sigma_m=0.25, sigma_growth_mps=0.8, max_sigma_m=2.0)
    est.observe(Pose2D(0, 0, 0), 0.0, 3.0, 10.0)
    assert est.predict(Pose2D(0, 0, 0), 10.5).sigma_m == pytest.approx(0.65)
    assert est.predict(Pose2D(0, 0, 0), 11.4) is not None
    assert est.predict(Pose2D(0, 0, 0), 11.6) is None           # past the TTL


def test_uncertainty_bound_refuses_before_ttl():
    est = CallerEstimate(ttl_s=5.0, base_sigma_m=0.25, sigma_growth_mps=0.8, max_sigma_m=1.0)
    est.observe(Pose2D(0, 0, 0), 0.0, 3.0, 0.0)
    assert est.predict(Pose2D(0, 0, 0), 0.9) is not None
    assert est.predict(Pose2D(0, 0, 0), 1.0) is None            # sigma 1.05 > 1.0


def test_observation_without_a_range_is_ignored():
    est = CallerEstimate()
    assert est.observe(Pose2D(0, 0, 0), 0.2, 0.0, 0.0) is False
    assert est.observe(Pose2D(0, 0, math.nan), 0.2, 2.0, 0.0) is False
    assert est.predict(Pose2D(0, 0, 0), 0.1) is None


def test_identity_gate_does_not_widen_with_time_by_default():
    est = CallerEstimate()
    est.observe(Pose2D(0, 0, 0), 0.0, 3.0, 0.0)
    here = Pose2D(0, 0, 0)
    assert est.consistent(here, 0.0, 3.5, 0.2, max_jump_m=1.2, memory_s=5.0)
    assert not est.consistent(here, 0.0, 5.5, 0.2, 1.2, 5.0)
    assert not est.consistent(here, 0.0, 5.5, 3.0, 1.2, 5.0)    # still someone else later
    assert est.consistent(here, 0.0, 5.5, 6.0, 1.2, 5.0)        # memory expired: no evidence


def test_invalid_config_refused():
    with pytest.raises(ValueError):
        CallerEstimate(ttl_s=0.0)


def test_default_window_is_bounded_by_uncertainty_before_the_ttl():
    """come_here_any.yaml defaults: sigma reaches 1.2 m at ~1.19 s, before the 1.5 s TTL."""
    est = CallerEstimate(ttl_s=1.5, base_sigma_m=0.25, sigma_growth_mps=0.8, max_sigma_m=1.2)
    est.observe(Pose2D(0, 0, 0), 0.0, 3.0, 0.0)
    assert est.predict(Pose2D(0, 0, 0), 1.15) is not None
    assert est.predict(Pose2D(0, 0, 0), 1.2) is None
