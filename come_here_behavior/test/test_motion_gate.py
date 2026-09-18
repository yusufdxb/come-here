"""Unit tests for MotionGate: bridge command validation, watchdog, e-stop latch.

Pure Python, no ROS: these run everywhere, including CI without unitree_api.
"""

import math

import pytest

from come_here_behavior.motion_gate import (
    MOVE,
    NONE,
    STOP,
    GateLimits,
    MotionGate,
)


def _limits(**overrides):
    values = dict(
        max_vx=0.7,
        max_yaw_rate=1.0,
        reject_vx_above=1.5,
        reject_yaw_rate_above=3.0,
        command_timeout_s=0.5,
    )
    values.update(overrides)
    return GateLimits(**values)


@pytest.fixture
def gate():
    return MotionGate(_limits())


# -- valid commands --

def test_forward_only_command_moves(gate):
    d = gate.on_command([0.6, 0.0], now_s=0.0)
    assert (d.action, d.vx, d.yaw_rate, d.reason) == (MOVE, 0.6, 0.0, 'ok')


def test_yaw_only_command_moves(gate):
    d = gate.on_command([0.0, -0.6], now_s=0.0)
    assert (d.action, d.vx, d.yaw_rate) == (MOVE, 0.0, -0.6)


def test_over_limit_forward_is_clamped(gate):
    d = gate.on_command([1.2, 0.0], now_s=0.0)
    assert (d.action, d.vx, d.reason) == (MOVE, 0.7, 'clamped')


def test_over_limit_yaw_is_clamped(gate):
    d = gate.on_command([0.0, -2.0], now_s=0.0)
    assert (d.action, d.yaw_rate, d.reason) == (MOVE, -1.0, 'clamped')


# -- rejected commands collapse to a stop --

@pytest.mark.parametrize('cmd', [[50.0, 0.0], [0.0, 9.0], [-1.6, 0.0]])
def test_absurd_magnitude_is_rejected(gate, cmd):
    d = gate.on_command(cmd, now_s=0.0)
    assert (d.action, d.reason) == (STOP, 'absurd')
    assert not gate.active


@pytest.mark.parametrize('cmd', [
    [math.nan, 0.0], [0.0, math.nan], [math.inf, 0.0], [0.0, -math.inf],
])
def test_non_finite_is_rejected(gate, cmd):
    d = gate.on_command(cmd, now_s=0.0)
    assert (d.action, d.reason) == (STOP, 'non_finite')


@pytest.mark.parametrize('cmd', [[], [0.6], [0.6, 0.0, 0.0], ['fast', 0.0]])
def test_malformed_array_is_rejected(gate, cmd):
    d = gate.on_command(cmd, now_s=0.0)
    assert (d.action, d.reason) == (STOP, 'malformed')


def test_combined_forward_and_yaw_is_rejected_by_default(gate):
    d = gate.on_command([0.6, 0.6], now_s=0.0)
    assert (d.action, d.reason) == (STOP, 'combined')
    assert not gate.active


def test_combined_allowed_when_configured():
    gate = MotionGate(_limits(allow_combined=True))
    d = gate.on_command([0.6, 0.6], now_s=0.0)
    assert (d.action, d.vx, d.yaw_rate) == (MOVE, 0.6, 0.6)


def test_zero_command_is_a_stop(gate):
    gate.on_command([0.6, 0.0], now_s=0.0)
    d = gate.on_command([0.0, 0.0], now_s=0.1)
    assert (d.action, d.reason) == (STOP, 'zero')
    assert gate.on_tick(now_s=0.15).action == NONE


def test_bad_command_while_moving_disarms_republisher(gate):
    gate.on_command([0.6, 0.0], now_s=0.0)
    assert gate.on_command([math.nan, 0.0], now_s=0.1).action == STOP
    assert gate.on_tick(now_s=0.15).action == NONE


# -- watchdog --

def test_tick_republishes_armed_command_within_timeout(gate):
    gate.on_command([0.6, 0.0], now_s=10.0)
    d = gate.on_tick(now_s=10.4)
    assert (d.action, d.vx, d.reason) == (MOVE, 0.6, 'republish')


def test_watchdog_stops_when_commands_stop_arriving(gate):
    gate.on_command([0.6, 0.0], now_s=10.0)
    d = gate.on_tick(now_s=10.51)
    assert (d.action, d.reason) == (STOP, 'watchdog')
    assert gate.on_tick(now_s=10.6).action == NONE


def test_fresh_command_resets_watchdog(gate):
    gate.on_command([0.6, 0.0], now_s=10.0)
    gate.on_command([0.6, 0.0], now_s=10.4)
    assert gate.on_tick(now_s=10.8).action == MOVE


def test_tick_does_nothing_when_idle(gate):
    assert gate.on_tick(now_s=0.0).action == NONE


# -- e-stop latch --

def test_estop_stops_and_blocks_commands(gate):
    gate.on_command([0.6, 0.0], now_s=0.0)
    d = gate.engage_estop()
    assert (d.action, d.reason) == (STOP, 'estop')
    assert gate.on_command([0.6, 0.0], now_s=0.1).action == NONE
    assert gate.on_tick(now_s=0.2).action == NONE
    assert gate.estopped


def test_estop_stays_latched_until_released(gate):
    gate.engage_estop()
    for t in range(20):
        assert gate.on_command([0.6, 0.0], now_s=float(t)).action == NONE
    assert gate.estopped


def test_release_requires_zero_command_before_motion(gate):
    gate.engage_estop()
    gate.release_estop()
    assert not gate.estopped
    d = gate.on_command([0.6, 0.0], now_s=1.0)
    assert (d.action, d.reason) == (STOP, 'rearm_required')
    assert gate.on_command([0.0, 0.0], now_s=1.1).action == STOP
    assert gate.on_command([0.6, 0.0], now_s=1.2).action == MOVE


def test_zero_sent_during_estop_does_not_rearm(gate):
    gate.engage_estop()
    gate.on_command([0.0, 0.0], now_s=0.5)
    gate.release_estop()
    assert gate.on_command([0.6, 0.0], now_s=1.0).reason == 'rearm_required'


def test_engaging_estop_twice_stops_both_times(gate):
    assert gate.engage_estop().action == STOP
    assert gate.engage_estop().action == STOP


# -- configuration --

@pytest.mark.parametrize('field,value', [
    ('max_vx', 0.0), ('max_vx', -1.0), ('max_yaw_rate', math.nan),
    ('command_timeout_s', 0.0), ('reject_vx_above', math.inf),
])
def test_invalid_limits_raise(field, value):
    with pytest.raises(ValueError):
        MotionGate(_limits(**{field: value}))


def test_reject_threshold_below_clamp_limit_raises():
    with pytest.raises(ValueError):
        MotionGate(_limits(max_vx=1.0, reject_vx_above=0.5))


# -- inhibit (unverified motion mode) --

def test_inhibit_stops_and_blocks_motion(gate):
    gate.on_command([0.6, 0.0], now_s=0.0)
    d = gate.inhibit('motion_mode_unverified')
    assert (d.action, d.reason) == (STOP, 'inhibited:motion_mode_unverified')
    assert gate.on_command([0.6, 0.0], now_s=0.1).action == STOP
    assert gate.on_tick(now_s=0.2).action == NONE
    assert gate.inhibited


def test_clearing_inhibit_still_requires_a_zero_command(gate):
    gate.inhibit('motion_mode_unverified')
    gate.inhibit(None)
    assert not gate.inhibited
    assert gate.on_command([0.6, 0.0], now_s=1.0).reason == 'rearm_required'
    gate.on_command([0.0, 0.0], now_s=1.1)
    assert gate.on_command([0.6, 0.0], now_s=1.2).action == MOVE


def test_estop_and_inhibit_are_independent(gate):
    gate.inhibit('motion_mode')
    gate.engage_estop()
    gate.inhibit(None)
    assert gate.on_command([0.0, 0.0], now_s=0.0).action == NONE  # still e-stopped
