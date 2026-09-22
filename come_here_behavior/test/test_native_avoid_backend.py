"""NativeAvoidBackend sequencing and NativeMotionGate rules (pure Python, no ROS).

These prove request formation and ordering against the Unitree SDK source
(unitree_sdk2py 1.0.1: sport_client.py FreeAvoid 2048 {"data"}, Move 1008
{"x","y","z"}; obstacles_avoid_client.py SwitchSet 1001 {"enable"}, SwitchGet
1002, Move 1003 {"x","y","yaw","mode":0} noreply, UseRemoteCommandFromApi 1004
{"is_remote_commands_from_api"}). They do NOT prove what the robot does with them.
"""

import json
import math

import pytest

from come_here_behavior.come_here_fsm import FsmConfig
from come_here_behavior.motion_gate import GateLimits, MotionGate
from come_here_behavior.native_avoid_backend import (
    BACKEND_FREEAVOID,
    BACKEND_OBSTACLES_AVOID,
    DISABLED,
    ENABLED,
    ENABLING,
    FAILED,
    MOVE,
    NONE,
    OBSTACLES_AVOID,
    SPORT,
    SPORT_API_SWITCHAVOIDMODE,
    STOP,
    NativeAvoidBackend,
    NativeGateLimits,
    NativeMotionGate,
    move_call,
)


def enable_live(backend, replies, now=0.0):
    """Drive the enable sequence, answering each call from ``replies`` (label -> (code, data))."""
    sent = []
    calls = backend.begin_enable(now)
    while calls:
        call = calls[0]
        sent.append(call)
        rid = backend.new_request_id()
        backend.on_published(rid, now)
        if call.label not in replies:
            break
        code, data = replies[call.label]
        calls = backend.on_response(call.service, rid, call.api_id, code, data, now)
    return sent


OA_OK = {
    'api_version': (0, '"1.0.0.2"'),
    'switch_get_initial': (0, '{"enable":false}'),
    'switch_set_on': (0, '{}'),
    'switch_get_verify': (0, '{"enable":true}'),
    'take_api_control': (0, '{}'),
}
FA_OK = {'api_version': (0, '"1.0.0.1"'), 'free_avoid_on': (0, '{}')}


# -- request formation ------------------------------------------------------

def test_freeavoid_enable_sends_sport_2048_data_true():
    b = NativeAvoidBackend(BACKEND_FREEAVOID)
    sent = enable_live(b, FA_OK)
    assert [(c.service, c.api_id) for c in sent] == [(SPORT, 1), (SPORT, 2048)]
    assert json.loads(sent[1].parameter_json()) == {'data': True}
    assert b.state == ENABLED
    st = b.status()
    assert st.enable_result == 'ok' and st.server_version == '1.0.0.1'
    assert st.api_version_match is True and st.api_control_taken is False


def test_obstacles_avoid_enable_matches_the_official_example_order():
    b = NativeAvoidBackend(BACKEND_OBSTACLES_AVOID)
    sent = enable_live(b, OA_OK)
    assert [(c.api_id, c.label) for c in sent] == [
        (1, 'api_version'), (1002, 'switch_get_initial'), (1001, 'switch_set_on'),
        (1002, 'switch_get_verify'), (1004, 'take_api_control')]
    assert all(c.service == OBSTACLES_AVOID and not c.noreply for c in sent)
    assert json.loads(sent[2].parameter_json()) == {'enable': True}
    assert json.loads(sent[4].parameter_json()) == {'is_remote_commands_from_api': True}
    assert b.enabled and b.api_control_taken and b.status().initial_switch is False


def test_move_requests_match_the_sdk_shapes():
    fa = move_call(BACKEND_FREEAVOID, 0.6, 0.0, 0.0)
    assert (fa.service, fa.api_id, fa.noreply) == (SPORT, 1008, False)
    assert json.loads(fa.parameter_json()) == {'x': 0.6, 'y': 0.0, 'z': 0.0}
    oa = move_call(BACKEND_OBSTACLES_AVOID, 0.5, 0.1, -0.2)
    assert (oa.service, oa.api_id, oa.noreply) == (OBSTACLES_AVOID, 1003, True)
    assert json.loads(oa.parameter_json()) == {'x': 0.5, 'y': 0.1, 'yaw': -0.2, 'mode': 0}
    with pytest.raises(ValueError):
        move_call(BACKEND_FREEAVOID, math.nan, 0.0, 0.0)


def test_switch_avoid_mode_toggle_is_never_sent():
    for backend, replies in ((BACKEND_FREEAVOID, FA_OK), (BACKEND_OBSTACLES_AVOID, OA_OK)):
        b = NativeAvoidBackend(backend)
        calls = enable_live(b, replies) + b.move(0.5, 0, 0) + b.stop_calls() + b.suspend()
        calls += b.release_calls()
        assert SPORT_API_SWITCHAVOIDMODE not in [c.api_id for c in calls]


# -- D / E: enable success and failure --------------------------------------

def test_no_motion_before_enabled():
    b = NativeAvoidBackend(BACKEND_FREEAVOID)
    assert b.move(0.6, 0.0, 0.0) == []
    b.begin_enable(0.0)
    assert b.state == ENABLING and b.move(0.6, 0.0, 0.0) == []


@pytest.mark.parametrize('label', ['free_avoid_on'])
def test_freeavoid_error_code_fails_closed(label):
    b = NativeAvoidBackend(BACKEND_FREEAVOID)
    enable_live(b, dict(FA_OK, **{label: (3203, '')}))
    assert b.state == FAILED and b.move(0.6, 0, 0) == []
    assert '3203' in b.status().failure


def test_obstacles_avoid_read_back_mismatch_fails():
    b = NativeAvoidBackend(BACKEND_OBSTACLES_AVOID)
    enable_live(b, dict(OA_OK, switch_get_verify=(0, '{"enable":false}')))
    assert b.state == FAILED
    assert b.move(0.5, 0, 0) == []


def test_enable_step_timeout_fails():
    b = NativeAvoidBackend(BACKEND_FREEAVOID, response_timeout_s=1.0)
    enable_live(b, {'api_version': (0, '"1.0.0.1"')}, now=0.0)   # FreeAvoid never answered
    b.tick(0.5)
    assert b.state == ENABLING
    b.tick(1.2)
    assert b.state == FAILED and 'no reply' in b.status().failure


def test_unanswered_version_query_moves_on_to_the_next_step():
    b = NativeAvoidBackend(BACKEND_FREEAVOID, response_timeout_s=1.0)
    b.begin_enable(0.0)
    b.on_published(b.new_request_id(), 0.0)
    nxt = b.tick(1.5)
    assert [c.api_id for c in nxt] == [2048] and b.state == ENABLING


def test_version_query_failure_is_not_fatal_but_recorded():
    b = NativeAvoidBackend(BACKEND_FREEAVOID)
    enable_live(b, dict(FA_OK, api_version=(3203, '')))
    assert b.enabled and b.status().server_version is None


def test_version_mismatch_is_recorded():
    b = NativeAvoidBackend(BACKEND_FREEAVOID)
    enable_live(b, dict(FA_OK, api_version=(0, '"1.0.0.9"')))
    assert b.enabled and b.status().api_version_match is False


def test_reply_for_another_request_is_ignored():
    b = NativeAvoidBackend(BACKEND_FREEAVOID)
    call = b.begin_enable(0.0)[0]
    rid = b.new_request_id()
    b.on_published(rid, 0.0)
    assert b.on_response(SPORT, rid + 7, call.api_id, 0, '"x"', 0.0) == []
    assert b.on_response(OBSTACLES_AVOID, rid, call.api_id, 0, '"x"', 0.0) == []
    assert b.state == ENABLING


def test_failed_is_sticky():
    b = NativeAvoidBackend(BACKEND_FREEAVOID)
    enable_live(b, dict(FA_OK, free_avoid_on=(1, '')))
    assert b.begin_enable(1.0) == [] and b.state == FAILED
    b.suspend()
    b.release_calls()
    assert b.state == FAILED


def test_dry_run_simulation_is_labelled_as_such():
    b = NativeAvoidBackend(BACKEND_OBSTACLES_AVOID, simulate_responses=True)
    calls = b.begin_enable(0.0)
    while calls:
        b.on_published(b.new_request_id(), 0.0)
        calls = b.simulated_reply(0.0)
    st = b.status()
    assert st.state == ENABLED and st.enable_result == 'dry_run_simulated' and st.simulated
    assert st.api_version_match is None      # nothing was compared against a robot


def test_live_backend_never_simulates():
    b = NativeAvoidBackend(BACKEND_FREEAVOID)
    b.begin_enable(0.0)
    b.on_published(b.new_request_id(), 0.0)
    assert b.simulated_reply(0.0) == [] and b.state == ENABLING


# -- F / G / K: disable, stop, release ---------------------------------------

def test_freeavoid_stop_is_stopmove_and_release_disables_freeavoid():
    b = NativeAvoidBackend(BACKEND_FREEAVOID)
    enable_live(b, FA_OK)
    assert [(c.service, c.api_id) for c in b.stop_calls()] == [(SPORT, 1003)]
    rel = b.release_calls()
    assert [c.label for c in rel] == ['stop_move', 'free_avoid_off']
    assert json.loads(rel[1].parameter_json()) == {'data': False}
    assert b.state == DISABLED


def test_obstacles_avoid_stop_zeroes_native_move_then_stopmove():
    b = NativeAvoidBackend(BACKEND_OBSTACLES_AVOID)
    enable_live(b, OA_OK)
    stop = b.stop_calls()
    assert [c.label for c in stop] == ['oa_move', 'stop_move']
    assert json.loads(stop[0].parameter_json()) == {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'mode': 0}


def test_obstacles_avoid_release_gives_control_back_and_restores_the_switch():
    b = NativeAvoidBackend(BACKEND_OBSTACLES_AVOID)
    enable_live(b, dict(OA_OK, switch_get_initial=(0, '{"enable":false}')))
    rel = b.release_calls()
    assert [c.label for c in rel] == ['oa_move', 'stop_move', 'release_api_control',
                                      'switch_restore']
    assert json.loads(rel[2].parameter_json()) == {'is_remote_commands_from_api': False}
    assert json.loads(rel[3].parameter_json()) == {'enable': False}
    st = b.status()
    assert st.api_control_released is True and st.api_control_taken is False


def test_suspend_releases_api_control_and_requires_re_enable():
    b = NativeAvoidBackend(BACKEND_OBSTACLES_AVOID)
    enable_live(b, OA_OK)
    calls = b.suspend()
    assert [c.label for c in calls] == ['oa_move', 'stop_move', 'release_api_control']
    assert b.state == DISABLED and b.move(0.5, 0, 0) == []
    assert enable_live(b, OA_OK)[-1].label == 'take_api_control' and b.enabled


def test_release_without_enable_sends_only_stops():
    b = NativeAvoidBackend(BACKEND_OBSTACLES_AVOID)
    assert [c.label for c in b.release_calls()] == ['oa_move', 'stop_move']
    b = NativeAvoidBackend(BACKEND_FREEAVOID)
    assert [c.label for c in b.release_calls()] == ['stop_move']


def test_unknown_backend_refused():
    with pytest.raises(ValueError):
        NativeAvoidBackend('lidar_planner')


# -- NativeMotionGate --------------------------------------------------------

def gate(**kw):
    base = dict(max_vx=0.7, max_vy=0.3, max_yaw_rate=1.0, reject_vx_above=1.5,
                reject_vy_above=1.0, reject_yaw_rate_above=3.0, command_timeout_s=0.5)
    base.update(kw)
    return NativeMotionGate(NativeGateLimits(**base))


def test_native_gate_rejects_legacy_two_element_commands():
    g = gate()
    d = g.on_command([0.6, 0.0], 0.0)
    assert d.action == STOP and d.reason == 'malformed'


def test_legacy_gate_rejects_three_element_any_commands():
    g = MotionGate(GateLimits(0.7, 1.0, 1.5, 3.0, 0.5))
    d = g.on_command([0.6, 0.0, 0.0], 0.0)
    assert d.action == 'stop' and d.reason == 'malformed'


def test_native_gate_single_axis_by_default():
    g = gate()
    assert g.on_command([0.6, 0.0, 0.0], 0.0).action == MOVE
    assert g.on_command([0.6, 0.0, 0.5], 0.0).reason == 'combined'
    assert g.on_command([0.0, 0.2, 0.0], 0.0).reason == 'lateral'
    assert g.on_command([0.0, 0.0, 0.5], 0.0).action == MOVE


def test_native_gate_lateral_and_combined_need_explicit_flags():
    g = gate(allow_lateral=True)
    assert g.on_command([0.5, 0.2, 0.0], 0.0).action == MOVE
    assert g.on_command([0.5, 0.2, 0.3], 0.0).reason == 'combined'
    g = gate(allow_lateral=True, allow_combined=True)
    d = g.on_command([0.5, 0.5, 0.3], 0.0)
    assert d.action == MOVE and d.vy == 0.3 and d.reason == 'clamped'


def test_native_gate_watchdog_estop_and_rearm():
    g = gate()
    g.on_command([0.6, 0.0, 0.0], 0.0)
    assert g.on_tick(0.3).action == MOVE
    assert g.on_tick(0.6).reason == 'watchdog'
    g.on_command([0.6, 0.0, 0.0], 1.0)
    g.engage_estop()
    assert g.on_command([0.6, 0.0, 0.0], 1.1).action == NONE
    g.release_estop()
    assert g.on_command([0.6, 0.0, 0.0], 1.2).reason == 'rearm_required'
    assert g.on_command([0.0, 0.0, 0.0], 1.3).reason == 'zero'
    assert g.on_command([0.6, 0.0, 0.0], 1.4).action == MOVE


def test_native_gate_inhibits_stack_and_mode_clear_keeps_native_reasons():
    g = gate()
    g.inhibit('motion_mode_unverified')
    g.set_inhibit('native_live_motion_not_cleared', True)
    g.inhibit(None)                     # the legacy mode check clears only its own reason
    assert g.inhibited and g.inhibit_reason == 'native_live_motion_not_cleared'
    assert g.on_command([0.6, 0.0, 0.0], 0.0).action == STOP


@pytest.mark.parametrize('cmd', [[math.nan, 0, 0], [0, math.inf, 0], [2.0, 0, 0], [0, 0, 5.0],
                                 [0, 1.5, 0], ['x', 0, 0]])
def test_native_gate_rejects_bad_values(cmd):
    d = gate(allow_lateral=True).on_command(cmd, 0.0)
    assert d.action == STOP and d.reason in ('non_finite', 'absurd', 'malformed')


def test_legacy_defaults_still_forbid_combined_motion():
    """Regression: nothing in ANY changed the legacy gate or FSM defaults."""
    assert GateLimits(0.7, 1.0, 1.5, 3.0, 0.5).allow_combined is False
    assert FsmConfig().walk_budget_arrives is False


# -- review 2026-09-21: a lost or late reply must still be released / restored --

def _publish_steps(b, upto_label, now=0.0):
    """Answer every step before ``upto_label`` with OA_OK/FA_OK; publish that one unanswered."""
    replies = OA_OK if b.backend == BACKEND_OBSTACLES_AVOID else FA_OK
    calls = b.begin_enable(now)
    while calls:
        call = calls[0]
        rid = b.new_request_id()
        b.on_published(rid, now)
        if call.label == upto_label:
            return call, rid
        code, data = replies[call.label]
        calls = b.on_response(call.service, rid, call.api_id, code, data, now)
    raise AssertionError(f'{upto_label} never sent')


def test_unanswered_take_api_control_is_released_after_timeout():
    b = NativeAvoidBackend(BACKEND_OBSTACLES_AVOID, response_timeout_s=1.0)
    _publish_steps(b, 'take_api_control')
    b.tick(2.0)
    assert b.state == FAILED
    labels = [c.label for c in b.release_calls()]
    assert 'release_api_control' in labels and 'switch_restore' in labels


def test_estop_during_take_api_control_releases_and_late_reply_changes_nothing():
    b = NativeAvoidBackend(BACKEND_OBSTACLES_AVOID)
    call, rid = _publish_steps(b, 'take_api_control')
    assert 'release_api_control' in [c.label for c in b.suspend()]
    assert b.on_response(call.service, rid, call.api_id, 0, '{}', 0.1) == []
    assert not b.enabled


def test_interrupted_switch_set_restores_to_the_state_before_we_touched_it():
    b = NativeAvoidBackend(BACKEND_OBSTACLES_AVOID)
    _publish_steps(b, 'switch_set_on')          # initial read: enable false
    b.suspend()
    # Re-enable: the robot now reads enable:true (our unanswered SwitchSet applied)
    enable_live(b, dict(OA_OK, switch_get_initial=(0, '{"enable":true}')))
    restore = [c for c in b.release_calls() if c.label == 'switch_restore']
    assert restore and restore[0].params == {'enable': False}


def test_unanswered_freeavoid_is_turned_off_at_shutdown():
    b = NativeAvoidBackend(BACKEND_FREEAVOID, response_timeout_s=1.0)
    _publish_steps(b, 'free_avoid_on')
    b.tick(2.0)
    assert b.state == FAILED
    assert 'free_avoid_off' in [c.label for c in b.release_calls()]
