"""Closed-loop turn: reaches the target on odometry, stops on stale odom, falls back timed."""

import math

import pytest

from come_here_behavior.rotate_controller import RotateController


def simulate(ctrl, rate_actual, dt=0.05, yaw0=0.0, lag_steps=0):
    """Integrate the commanded rate into a fake odometry yaw; return (steps, yaw)."""
    yaw = yaw0
    now = 0.0
    steps = []
    for _ in range(400):
        step = ctrl.step(now, yaw, now)
        steps.append(step)
        if step.done:
            break
        yaw = math.atan2(math.sin(yaw + rate_actual * math.copysign(1.0, step.yaw_rate) * dt),
                         math.cos(yaw + rate_actual * math.copysign(1.0, step.yaw_rate) * dt))
        now += dt
    return steps, yaw


@pytest.mark.parametrize('target_deg', [90, -90, 45, -20, 170])
def test_reaches_the_target_on_odometry(target_deg):
    target = math.radians(target_deg)
    ctrl = RotateController(target, yaw_rate=1.0, start_s=0.0, start_yaw=0.3, deadband_rad=0.1)
    steps, yaw = simulate(ctrl, rate_actual=1.0, yaw0=0.3)
    assert steps[-1].reason == 'reached'
    assert steps[-1].yaw_rate == 0.0
    final_err = math.atan2(math.sin(yaw - (0.3 + target)), math.cos(yaw - (0.3 + target)))
    assert abs(final_err) <= 0.1 + 0.05 * 1.0 + 1e-9  # deadband plus one control step


def test_turns_the_commanded_direction():
    left = RotateController(math.radians(90), 1.0, 0.0, 0.0)
    right = RotateController(math.radians(-90), 1.0, 0.0, 0.0)
    assert left.step(0.0, 0.0, 0.0).yaw_rate > 0
    assert right.step(0.0, 0.0, 0.0).yaw_rate < 0


def test_a_target_near_behind_prefers_the_counter_clockwise_long_way():
    ctrl = RotateController(math.radians(-160), 1.0, 0.0, 0.0, prefer_ccw_beyond_rad=2.6)
    assert ctrl.sign == 1.0
    assert ctrl.magnitude_rad == pytest.approx(math.radians(200))
    steps, yaw = simulate(ctrl, rate_actual=1.0)
    assert steps[-1].reason == 'reached'
    assert abs(math.degrees(yaw) - (-160)) <= 8.0


def test_yaw_wraparound_does_not_confuse_progress():
    ctrl = RotateController(math.radians(60), 1.0, 0.0, start_yaw=math.radians(170), deadband_rad=0.05)
    steps, yaw = simulate(ctrl, rate_actual=1.0, yaw0=math.radians(170))
    assert steps[-1].reason == 'reached'
    assert abs(math.degrees(yaw) - (-130)) <= 6.0


def test_a_slower_robot_still_reaches_and_a_faster_one_overshoots_once():
    slow = RotateController(math.radians(90), 1.0, 0.0, 0.0, deadband_rad=0.05)
    steps, _ = simulate(slow, rate_actual=0.4)
    assert steps[-1].reason == 'reached'
    fast = RotateController(math.radians(20), 1.0, 0.0, 0.0, deadband_rad=0.02)
    steps, _ = simulate(fast, rate_actual=3.0, dt=0.2)
    assert steps[-1].reason == 'overshoot'
    assert steps[-1].yaw_rate == 0.0


def test_stale_odometry_stops_the_turn():
    ctrl = RotateController(math.radians(90), 1.0, 0.0, 0.0, odom_max_age_s=0.5)
    assert not ctrl.step(0.1, 0.05, 0.1).done
    step = ctrl.step(1.0, 0.05, 0.1)           # last yaw is 0.9 s old
    assert step.done and step.reason == 'odom_stale' and step.yaw_rate == 0.0
    step = ctrl.step(1.5, 0.05, 1.5)           # stays done
    assert step.done


def test_missing_odometry_stops_the_turn_too():
    ctrl = RotateController(math.radians(90), 1.0, 0.0, 0.0)
    assert ctrl.step(0.1, None, None).reason == 'odom_stale'


def test_timeout_bounds_a_turn_that_never_reaches():
    ctrl = RotateController(math.radians(90), 1.0, 0.0, 0.0, timeout_s=2.0)
    now = 0.0
    while True:
        step = ctrl.step(now, 0.0, now)        # robot never moves
        if step.done:
            break
        now += 0.05
    assert step.reason == 'timeout' and now <= 2.1


def test_falls_back_to_a_timed_turn_without_a_start_yaw():
    ctrl = RotateController(math.radians(90), 1.0, 0.0, start_yaw=None, fallback_deg_per_sec=90.0)
    assert not ctrl.closed_loop
    assert ctrl.timed_duration_s == pytest.approx(1.0)
    assert ctrl.step(0.5).yaw_rate == pytest.approx(1.0)
    step = ctrl.step(1.0)
    assert step.done and step.reason == 'timed'


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        RotateController(4.0, 1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        RotateController(1.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        RotateController(float('nan'), 1.0, 0.0, 0.0)
