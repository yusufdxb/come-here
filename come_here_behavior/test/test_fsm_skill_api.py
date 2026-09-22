"""Stepwise skill interface v2 of ComeHereFsm (fake time, professor_demo parameters).

Invariants under test: one result per request, naming its goal_id and request_id; one
active request; a step that depends on an earlier one names it (localization ->
orient, acquisition -> approach, arrival -> sit) and the handoff is used once, expires,
and is void after any motion or a new goal; approach_caller never sits and ends
standing and stopped; only the sit skill sits; invalid, stale, duplicate or
out-of-order requests are rejected without any motion command; cancel, timeouts and
e-stop end a skill through the existing stop paths; the baseline path is unaffected.
"""

import dataclasses
import math
import os
import random
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import golden_baseline as gb  # noqa: E402
from come_here_behavior.come_here_fsm import (  # noqa: E402
    GOAL_ID_MEMORY, SKILL_ACQUISITION_MAX_S, SKILL_ARRIVAL_VALID_S, SKILL_POSTURE_PHASES,
    ComeHereFsm, State,
)


class SkillDriver(gb.Driver):
    """The golden-trace driver, recording skill results instead of refusing them."""

    def __init__(self, world, config=None, **kw):
        super().__init__(world, **kw)
        if config is not None:
            self.fsm = ComeHereFsm(config)
        self.results = []
        self.frames = []
        self.active = []
        self.seq = {}
        self.n = 0

    def rec(self, name, cmds):
        if cmds.velocity is not None:
            self.cmd = cmds.velocity
        if cmds.rotate_rad is not None:
            self.pending_rotate = (cmds.rotate_rad, self.t + 1.0)
        self.frames.append((self.t, name, cmds))
        self.active.append(self.fsm._skill)      # the skill running after this call
        self.results.extend(cmds.skill_results)
        if cmds.face_request and self.face:
            self.rec('face', self.fsm.on_face_result(True, self.t, 0.5))
        return cmds

    def request(self, skill, goal='g1', rid=None, seq=None, v=2, **args):
        self.n += 1
        rid = rid or f'{goal}-r{self.n}'
        if seq is None:
            seq = self.seq.get(goal, 0) + 1
        self.seq[goal] = max(self.seq.get(goal, 0), seq)
        req = {'v': v, 'goal_id': goal, 'request_id': rid, 'seq': seq, 'skill': skill,
               'args': args}
        self.rec('request', self.fsm.on_skill_request(req, self.t))
        return rid

    def raw(self, req):
        return self.rec('request', self.fsm.on_skill_request(req, self.t))

    def result(self, rid):
        found = [r for r in self.results if r['request_id'] == rid]
        assert len(found) == 1, f'{rid}: expected one result, got {found}'
        return found[0]

    def run_until(self, rid, limit=40.0):
        for _ in range(round(limit / gb.TICK)):
            if any(r['request_id'] == rid for r in self.results):
                return self.result(rid)
            self.run(gb.TICK)
        raise AssertionError(f'{rid}: no result within {limit} s')

    def summaries(self):
        return [c.trial_summary for _, _, c in self.frames if c.trial_summary]

    def velocities(self):
        return [c.velocity for _, _, c in self.frames if c.velocity is not None]

    def index_of_result(self, rid):
        return next(i for i, (_, _, c) in enumerate(self.frames)
                    if any(r['request_id'] == rid for r in c.skill_results))


def _no_motion(cmds):
    return (cmds.velocity is None and cmds.rotate_rad is None and not cmds.sit
            and not cmds.stand and cmds.trial_summary is None)


def _voice(d, conf=0.9, offset=0.0):
    d.direction(d.world.bearing() + offset, conf)


def _chain_to_acquired(d):
    """localize -> orient -> acquire, each succeeding; returns the acquisition id."""
    _voice(d)
    loc = d.request('localize_caller')
    assert d.run_until(loc)['status'] == 'succeeded'
    ori = d.request('orient_to_caller', localization=loc)
    assert d.run_until(ori)['status'] == 'succeeded'
    acq = d.request('acquire_caller')
    assert d.run_until(acq)['status'] == 'succeeded', d.result(acq)
    return acq


# -- the full decomposed sequence --

def test_full_sequence_approach_ends_standing_then_a_separate_sit():
    d = SkillDriver(gb.World(1.2, 2.6))
    acq = _chain_to_acquired(d)
    app = d.request('approach_caller', acquisition=acq)
    r = d.run_until(app)
    assert r['status'] == 'succeeded' and r['reason'].startswith('arrived')
    assert r['data']['posture'] == 'standing'
    assert not any(c.sit for _, _, c in d.frames), 'approach must never sit'
    assert d.velocities()[-1] == (0.0, 0.0)
    assert d.fsm.state == State.IDLE
    # Standing and stopped, the robot waits for the next decision; nothing happens.
    n = len(d.frames)
    d.run(3)
    assert not any(c.sit or (c.velocity or (0, 0)) != (0, 0) or c.rotate_rad is not None
                   for _, _, c in d.frames[n:])
    sit = d.request('sit', arrival=app)
    s = d.run_until(sit)
    assert s['status'] == 'succeeded' and s['reason'] == 'sit_commanded'
    assert s['data']['posture'] == 'unverified'
    sits = [i for i, (_, _, c) in enumerate(d.frames) if c.sit]
    assert len(sits) == 1 and sits[0] > d.index_of_result(app)
    assert not any(c.velocity not in (None, (0.0, 0.0)) for _, _, c in d.frames[n:])
    assert d.fsm.state == State.SIT_AND_IDENTIFY
    # Seated: every request that could move is refused until the operator reset.
    rid = d.request('localize_caller')
    assert d.result(rid)['reason'] == 'not_idle'
    d.rec('reset', d.fsm.on_reset(d.t))
    d.run(2)
    assert d.fsm.state == State.IDLE
    assert any(c.stand for _, _, c in d.frames)


def test_every_result_names_goal_and_request():
    d = SkillDriver(gb.World(0.05, 2.4))
    acq = _chain_to_acquired(d)
    d.run_until(d.request('approach_caller', acquisition=acq))
    for r in d.results:
        assert r['goal_id'] == 'g1' and r['request_id'].startswith('g1-r')
        assert set(r) == {'v', 'goal_id', 'request_id', 'skill', 'status', 'reason', 'data'}


# -- localize_caller --

def test_localize_reports_bearing_confidence_and_age_and_never_moves():
    d = SkillDriver(gb.World(1.2, 2.4))
    _voice(d, 0.8)
    d.run(0.5)
    loc = d.request('localize_caller')
    r = d.run_until(loc)
    assert r['status'] == 'succeeded' and r['reason'] == 'localized'
    assert r['data']['bearing_rad'] == pytest.approx(1.2, abs=1e-3)
    assert r['data']['confidence'] == pytest.approx(0.8)
    assert 0.4 <= r['data']['age_s'] <= 0.7
    assert all(v == (0.0, 0.0) for v in d.velocities())
    assert not any(c.rotate_rad is not None for _, _, c in d.frames)


def test_localize_without_a_bearing_fails_no_direction():
    d = SkillDriver(gb.World(1.2, 2.4))
    r = d.run_until(d.request('localize_caller'))
    assert r['status'] == 'failed' and r['reason'] == 'no_direction'
    assert r['data']['bearing_rad'] is None


def test_localize_low_confidence_fails_and_reports_the_bearing():
    d = SkillDriver(gb.World(1.2, 2.4))
    _voice(d, 0.2)
    r = d.run_until(d.request('localize_caller'))
    assert r['status'] == 'failed' and r['reason'] == 'low_confidence'
    assert r['data']['confidence'] == pytest.approx(0.2)
    assert not any(c.rotate_rad is not None for _, _, c in d.frames)


def test_localize_stale_bearing_fails():
    from come_here_behavior.come_here_fsm import SKILL_LOCALIZATION_MAX_AGE_S
    d = SkillDriver(gb.World(1.2, 2.4))
    _voice(d)
    d.run(SKILL_LOCALIZATION_MAX_AGE_S + 0.5)
    assert d.run_until(d.request('localize_caller'))['reason'] == 'no_direction'


def test_a_bearing_from_before_a_turn_never_localizes_again():
    d = SkillDriver(gb.World(1.2, 2.6))
    _voice(d)
    loc = d.request('localize_caller')
    d.run_until(loc)
    d.run_until(d.request('orient_to_caller', localization=loc))
    # Same (pre-turn) bearing still within the age limit: void after the turn.
    r = d.run_until(d.request('localize_caller'))
    assert r['status'] == 'failed' and r['reason'] == 'no_direction'
    _voice(d)                                   # a new utterance after the turn
    assert d.run_until(d.request('localize_caller'))['status'] == 'succeeded'


# -- orient_to_caller --

def test_orient_turns_by_the_localized_bearing_even_if_a_newer_one_arrives():
    d = SkillDriver(gb.World(1.2, 2.6))
    _voice(d)
    loc = d.request('localize_caller')
    d.run_until(loc)
    d.direction(-0.9, 0.95)                     # a later, different bearing
    ori = d.request('orient_to_caller', localization=loc)
    r = d.run_until(ori)
    assert r['status'] == 'succeeded' and r['reason'] == 'turn_complete'
    rot = [c.rotate_rad for _, _, c in d.frames if c.rotate_rad is not None]
    assert rot == [pytest.approx(1.2, abs=1e-3)]
    assert all(v == (0.0, 0.0) for v in d.velocities()), 'a turn skill never walks'


def test_orient_not_needed_when_the_caller_is_ahead():
    d = SkillDriver(gb.World(0.05, 2.4))
    _voice(d)
    loc = d.request('localize_caller')
    d.run_until(loc)
    r = d.run_until(d.request('orient_to_caller', localization=loc))
    assert r['status'] == 'succeeded' and r['reason'] == 'turn_not_needed'
    assert not any(c.rotate_rad is not None for _, _, c in d.frames)


@pytest.mark.parametrize('case', ['none', 'wrong_id', 'reused', 'stale', 'failed_loc'])
def test_orient_needs_a_fresh_unused_localization(case):
    d = SkillDriver(gb.World(1.2, 2.6))
    if case == 'failed_loc':
        _voice(d, 0.1)
    else:
        _voice(d)
    loc = d.request('localize_caller')
    d.run_until(loc)
    expected = {'none': 'no_localization', 'wrong_id': 'no_localization',
                'reused': 'no_localization', 'stale': 'stale_localization',
                'failed_loc': 'no_localization'}[case]
    if case == 'reused':
        d.run_until(d.request('orient_to_caller', localization=loc))
    if case == 'stale':
        from come_here_behavior.come_here_fsm import SKILL_LOCALIZATION_MAX_AGE_S
        d.run(SKILL_LOCALIZATION_MAX_AGE_S + 0.2)
    n = len(d.frames)
    if case == 'none':
        rid = d.request('orient_to_caller', localization='never-issued')
    elif case == 'wrong_id':
        rid = d.request('orient_to_caller', localization='g1-r999')
    else:
        rid = d.request('orient_to_caller', localization=loc)
    assert d.result(rid)['status'] == 'rejected' and d.result(rid)['reason'] == expected
    assert all(_no_motion(c) for _, _, c in d.frames[n:])


def test_orient_without_rotate_result_fails():
    d = SkillDriver(gb.World(1.2, 2.4), rotate_result=False)
    _voice(d)
    loc = d.request('localize_caller')
    d.run_until(loc)
    r = d.run_until(d.request('orient_to_caller', localization=loc), limit=15)
    assert r['status'] == 'failed' and r['reason'] == 'turn_no_result'
    assert d.fsm.state == State.IDLE


# -- acquire_caller --

def test_acquire_never_moves_and_reports_geometry():
    d = SkillDriver(gb.World(0.1, 2.4))
    r = d.run_until(d.request('acquire_caller'))
    assert r['status'] == 'succeeded' and r['reason'] == 'acquired'
    for key in ('bearing_rad', 'confidence', 'distance_m', 'bbox_h_frac', 'age_s', 'track'):
        assert r['data'][key] is not None
    assert all(v == (0.0, 0.0) for v in d.velocities())
    assert not any(c.rotate_rad is not None for _, _, c in d.frames)


def test_acquire_nobody_times_out_without_search_or_relisten():
    d = SkillDriver(gb.World(0.1, 2.4, visible=False))
    r = d.run_until(d.request('acquire_caller'), limit=20)
    assert r['status'] == 'failed' and r['reason'] == 'acquire_timeout'
    assert not any(c.rotate_rad is not None for _, _, c in d.frames), 'no search turn'
    assert not any(c.say for _, _, c in d.frames), 'no relisten prompt'
    assert all(v == (0.0, 0.0) for v in d.velocities())


def test_acquisition_holds_the_gate_then_expires():
    d = SkillDriver(gb.World(0.1, 2.4))
    acq = d.request('acquire_caller')
    d.run_until(acq)
    d.run(0.5)
    gates = [c.gate for _, n, c in d.frames[-3:] if n == 'tick']
    assert gates and all(g is not None and g[1] > 0 for g in gates)
    d.run(SKILL_ACQUISITION_MAX_S)
    rid = d.request('approach_caller', acquisition=acq)
    assert d.result(rid)['reason'] in ('stale_acquisition', 'no_acquisition')
    assert d.frames[-1][2].velocity is None


def test_acquisition_survives_a_slow_planner_while_the_track_holds():
    d = SkillDriver(gb.World(0.1, 2.4))
    acq = d.request('acquire_caller')
    d.run_until(acq)
    d.run(6.0)                                   # a slow decision; the caller stays in view
    rid = d.request('approach_caller', acquisition=acq)
    assert d.fsm.state == State.ACQUIRE_PERSON, d.result(rid)


def test_acquisition_breaks_when_the_track_breaks():
    d = SkillDriver(gb.World(0.1, 2.4))
    acq = d.request('acquire_caller')
    d.run_until(acq)
    d.world.visible = False
    d.run(d.fsm.config.person_stale_timeout_s + 0.5)
    d.world.visible = True
    d.run(1.0)                                   # back in view, but the track broke
    rid = d.request('approach_caller', acquisition=acq)
    assert d.result(rid)['reason'] == 'no_acquisition'


# -- approach_caller --

def test_approach_needs_an_acquisition():
    d = SkillDriver(gb.World(0.05, 2.4))
    d.run(1)
    rid = d.request('approach_caller', acquisition='g1-r77')
    assert d.result(rid)['reason'] == 'no_acquisition'
    assert _no_motion(d.frames[-1][2])


def test_approach_refused_when_caller_left_view_after_acquisition():
    d = SkillDriver(gb.World(0.1, 2.4))
    acq = d.request('acquire_caller')
    d.run_until(acq)
    d.world.visible = False
    d.run(d.fsm.config.person_stale_timeout_s + 0.3)
    rid = d.request('approach_caller', acquisition=acq)
    assert d.result(rid)['reason'] in ('caller_not_in_view', 'stale_acquisition',
                                       'no_acquisition')
    assert _no_motion(d.frames[-1][2])


def test_approach_never_sits_even_when_left_running():
    d = SkillDriver(gb.World(0.05, 2.4))
    acq = d.request('acquire_caller')
    d.run_until(acq)
    app = d.request('approach_caller', acquisition=acq)
    d.run_until(app)
    d.run(20)
    assert not any(c.sit for _, _, c in d.frames)
    assert d.fsm.state == State.IDLE


def test_track_lost_mid_approach_stops_and_ends_the_step():
    d = SkillDriver(gb.World(0.05, 3.5))
    acq = d.request('acquire_caller')
    d.run_until(acq)
    app = d.request('approach_caller', acquisition=acq)
    d.run(2.0)
    assert any(v[0] > 0 for v in d.velocities()), 'walking'
    d.world.visible = False
    r = d.run_until(app, limit=5)
    assert r['status'] == 'failed' and r['reason'] == 'track_lost'
    assert d.velocities()[-1] == (0.0, 0.0)
    assert d.fsm.state == State.IDLE
    n = len(d.frames)
    d.world.visible = True
    d.run(3)
    assert all(_no_motion(c) or c.velocity in (None, (0.0, 0.0))
               for _, _, c in d.frames[n:]), 'no reacquisition without a request'
    # One bounded reacquisition is the supervisor's decision: it gates at the lost bearing.
    re = d.request('acquire_caller')
    r2 = d.run_until(re)
    assert r2['status'] == 'succeeded'
    assert r2['data']['gate_center_rad'] == pytest.approx(r['data']['final_bearing_rad']
                                                          or 0.0, abs=0.2)


@pytest.mark.parametrize('arrives', [True, False])
def test_walk_budget_follows_the_baseline_rule_and_is_labelled(arrives):
    cfg = dataclasses.replace(gb.demo_config(), walk_budget_arrives=arrives,
                              max_walk_distance_m=1.0)
    d = SkillDriver(gb.World(0.02, 4.0), config=cfg)
    acq = d.request('acquire_caller')
    d.run_until(acq)
    r = d.run_until(d.request('approach_caller', acquisition=acq))
    if arrives:
        assert r['status'] == 'succeeded' and r['reason'] == 'arrived_walk_budget'
    else:
        assert r['status'] == 'failed' and r['reason'] == 'walk_budget'
    assert r['data']['proximity_evidence'] is False
    assert not any(c.sit for _, _, c in d.frames)
    assert d.velocities()[-1] == (0.0, 0.0)


# -- sit --

def _arrived(d):
    acq = d.request('acquire_caller')
    d.run_until(acq)
    app = d.request('approach_caller', acquisition=acq)
    assert d.run_until(app)['status'] == 'succeeded'
    return app


def test_sit_needs_an_arrival():
    d = SkillDriver(gb.World(0.05, 2.4))
    rid = d.request('sit', arrival='g1-r5')
    assert d.result(rid)['reason'] == 'no_arrival'
    assert _no_motion(d.frames[-1][2])


def test_sit_after_a_failed_approach_is_refused():
    d = SkillDriver(gb.World(0.05, 3.5))
    acq = d.request('acquire_caller')
    d.run_until(acq)
    app = d.request('approach_caller', acquisition=acq)
    d.run(1.5)
    d.world.visible = False
    assert d.run_until(app)['status'] == 'failed'
    rid = d.request('sit', arrival=app)
    assert d.result(rid)['reason'] == 'no_arrival'


def test_stale_arrival_is_refused():
    d = SkillDriver(gb.World(0.05, 2.4))
    app = _arrived(d)
    d.run(SKILL_ARRIVAL_VALID_S + 0.5)
    rid = d.request('sit', arrival=app)
    assert d.result(rid)['reason'] == 'stale_arrival'


def test_an_arrival_is_used_once():
    d = SkillDriver(gb.World(0.05, 2.4))
    app = _arrived(d)
    sit = d.request('sit', arrival=app)
    d.run_until(sit)
    d.rec('reset', d.fsm.on_reset(d.t))
    d.run(2)
    rid = d.request('sit', arrival=app)
    assert d.result(rid)['reason'] == 'no_arrival'


def test_cancel_between_arrival_and_sit_never_sits():
    d = SkillDriver(gb.World(0.05, 2.4))
    app = _arrived(d)
    sit = d.request('sit', arrival=app)
    d.run(0.3)                                  # still settling, Sit not sent
    c = d.request('cancel', request_id=sit)
    assert d.result(sit)['status'] == 'cancelled'
    assert d.result(c)['status'] == 'succeeded'
    d.run(5)
    assert not any(f.sit for _, _, f in d.frames)
    assert d.fsm.state == State.IDLE


@pytest.mark.parametrize('phase', ['sit', 'done'])
def test_cancel_once_sitting_is_refused(phase):
    assert phase in SKILL_POSTURE_PHASES
    d = SkillDriver(gb.World(0.05, 2.4))
    app = _arrived(d)
    sit = d.request('sit', arrival=app)
    if phase == 'sit':
        d.run(d.fsm.config.pre_sit_settle_s + 0.2)
        c = d.request('cancel', request_id=sit)
        assert d.result(c)['reason'] == 'posture_phase'
    else:
        d.run_until(sit)
        c = d.request('cancel', request_id=sit)
        assert d.result(c)['reason'] == 'not_active'


def test_estop_after_sit_blocks_motion_until_operator_reset():
    d = SkillDriver(gb.World(0.05, 2.4))
    app = _arrived(d)
    d.run_until(d.request('sit', arrival=app))
    d.rec('estop', d.fsm.on_estop(True, d.t))
    d.rec('estop', d.fsm.on_estop(False, d.t))
    assert d.fsm.state == State.IDLE
    _voice(d)
    loc = d.request('localize_caller')
    d.run_until(loc)
    rid = d.request('orient_to_caller', localization=loc)
    assert d.result(rid)['reason'] == 'seated'
    d.rec('estop', d.fsm.on_estop(True, d.t))
    d.rec('reset', d.fsm.on_reset(d.t))
    assert not d.frames[-1][2].stand, 'reset while e-stopped does nothing'
    d.rec('estop', d.fsm.on_estop(False, d.t))
    d.rec('reset', d.fsm.on_reset(d.t))
    assert d.frames[-1][2].stand, 'released: the reset stands the robot up'
    d.run(2)
    assert d.fsm.state == State.IDLE
    _voice(d)
    loc = d.request('localize_caller')
    d.run_until(loc)
    rid = d.request('orient_to_caller', localization=loc)
    assert d.run_until(rid)['status'] == 'succeeded'


# -- ask_caller_again --

def test_ask_again_speaks_and_only_a_later_bearing_localizes():
    d = SkillDriver(gb.World(1.2, 2.4))
    _voice(d)
    ask = d.request('ask_caller_again')
    r = d.result(ask)
    assert r['status'] == 'succeeded' and d.frames[-1][2].say
    assert all(v == (0.0, 0.0) for v in d.velocities())
    loc = d.request('localize_caller')
    d.run(2.0)
    _voice(d)                                   # the caller answers
    r = d.run_until(loc)
    assert r['status'] == 'succeeded'
    assert r['data']['age_s'] < 0.5


def test_ask_again_then_silence_fails_after_the_relisten_window():
    d = SkillDriver(gb.World(1.2, 2.4))
    _voice(d)
    d.request('ask_caller_again')
    loc = d.request('localize_caller')
    r = d.run_until(loc, limit=15)
    assert r['reason'] == 'no_direction'
    assert d.frames[-1][0] >= d.fsm.config.relisten_timeout_s - 0.2


# -- correlation: ids, sequence, goals --

def test_duplicate_request_id_is_rejected():
    d = SkillDriver(gb.World(0.05, 2.4))
    d.run_until(d.request('ask_caller_again', rid='x1'))
    cmds = d.raw({'v': 2, 'goal_id': 'g1', 'request_id': 'x1', 'seq': 9,
                  'skill': 'ask_caller_again', 'args': {}})
    assert [r['reason'] for r in cmds.skill_results] == ['duplicate_request_id']
    assert _no_motion(cmds) and cmds.say is None


def test_out_of_order_request_is_rejected():
    d = SkillDriver(gb.World(0.05, 2.4))
    d.run_until(d.request('ask_caller_again', seq=5))
    rid = d.request('ask_caller_again', seq=3)
    assert d.result(rid)['reason'] == 'out_of_order'


def test_a_new_goal_voids_the_previous_goals_handoffs_and_retires_it():
    d = SkillDriver(gb.World(0.05, 2.4))
    acq = d.request('acquire_caller', goal='g1')
    d.run_until(acq)
    rid = d.request('approach_caller', goal='g2', acquisition=acq)
    assert d.result(rid)['reason'] == 'no_acquisition'
    old = d.request('ask_caller_again', goal='g1')
    assert d.result(old)['reason'] == 'stale_goal'


def test_a_new_goal_cannot_start_while_another_goal_runs():
    d = SkillDriver(gb.World(0.1, 2.4, visible=False))
    active = d.request('acquire_caller', goal='g1')
    rid = d.request('acquire_caller', goal='g2')
    assert d.result(rid)['reason'] == 'not_idle'
    c = d.request('cancel', goal='g2', request_id=active)
    assert d.result(c)['reason'] == 'not_active'
    assert d.fsm.state == State.ACQUIRE_PERSON


def test_only_one_active_request():
    d = SkillDriver(gb.World(0.1, 2.4, visible=False))
    d.request('acquire_caller')
    rid = d.request('localize_caller')
    assert d.result(rid)['reason'] == 'not_idle'
    assert d.fsm.state == State.ACQUIRE_PERSON


def _base(**over):
    req = {'v': 2, 'goal_id': 'g', 'request_id': 'r', 'seq': 1,
           'skill': 'localize_caller', 'args': {}}
    req.update(over)
    return req


@pytest.mark.parametrize('req,reason', [
    (None, 'malformed'),
    ('text', 'malformed'),
    (_base(v=1), 'malformed'),
    (_base(goal_id=None), 'malformed'),
    (_base(goal_id=''), 'malformed'),
    (_base(goal_id='x' * 65), 'malformed'),
    (_base(request_id=None), 'malformed'),
    (_base(request_id='bad\nid'), 'malformed'),
    (_base(seq=0), 'malformed'),
    (_base(seq=True), 'malformed'),
    (_base(seq='1'), 'malformed'),
    (_base(args=[]), 'malformed'),
    (_base(skill='fly'), 'unknown_skill'),
    (_base(skill='turn_to_voice'), 'unknown_skill'),
    (_base(skill='approach_person'), 'unknown_skill'),
    (_base(args={'speed': 2.0}), 'bad_args'),
    (_base(skill='approach_caller', args={}), 'bad_args'),
    (_base(skill='approach_caller', args={'acquisition': 'a', 'arrival': 'sit'}),
     'bad_args'),
    (_base(skill='orient_to_caller', args={'localization': 1.2}), 'bad_args'),
    (_base(skill='sit', args={}), 'bad_args'),
])
def test_invalid_requests_are_rejected_without_motion(req, reason):
    d = SkillDriver(gb.World(0.05, 2.4))
    cmds = d.raw(req)
    assert _no_motion(cmds)
    assert [r['reason'] for r in cmds.skill_results] == [reason]
    assert cmds.skill_results[0]['status'] == 'rejected'
    assert d.fsm.state == State.IDLE


def test_requests_are_refused_while_estopped():
    d = SkillDriver(gb.World(0.05, 2.4))
    d.rec('estop', d.fsm.on_estop(True, d.t))
    n = len(d.frames)
    rid = d.request('acquire_caller')
    assert d.result(rid)['reason'] == 'estopped'
    assert all(_no_motion(c) for _, _, c in d.frames[n:])


def test_estop_during_approach_reports_cancelled_estop_and_voids_handoffs():
    d = SkillDriver(gb.World(0.05, 3.5))
    acq = d.request('acquire_caller')
    d.run_until(acq)
    app = d.request('approach_caller', acquisition=acq)
    d.run(1.5)
    d.rec('estop', d.fsm.on_estop(True, d.t))
    r = d.result(app)
    assert r['status'] == 'cancelled' and r['reason'] == 'estop'
    assert d.velocities()[-1] == (0.0, 0.0)
    d.rec('estop', d.fsm.on_estop(False, d.t))
    rid = d.request('sit', arrival=app)
    assert d.result(rid)['reason'] == 'no_arrival'


def test_cancel_mid_walk_stops_through_abort_and_latches_nothing():
    d = SkillDriver(gb.World(0.05, 3.5))
    acq = d.request('acquire_caller')
    d.run_until(acq)
    app = d.request('approach_caller', acquisition=acq)
    d.run(1.5)
    c = d.request('cancel', request_id=app)
    r = d.result(app)
    assert r['status'] == 'cancelled' and r['data']['cancel_request_id'] == c
    assert d.velocities()[-1] == (0.0, 0.0)
    assert not d.fsm.estopped
    assert d.fsm.state == State.IDLE


def test_a_wake_during_a_skill_session_is_ignored():
    d = SkillDriver(gb.World(0.05, 2.4))
    d.run_until(d.request('ask_caller_again'))
    cmds = d.rec('wake', d.fsm.on_wake('come here', d.t))
    assert _no_motion(cmds) and d.fsm.state == State.IDLE


def test_shutdown_during_a_skill_reports_failed():
    d = SkillDriver(gb.World(0.1, 2.4, visible=False))
    rid = d.request('acquire_caller')
    d.run(1)
    d.rec('shutdown', d.fsm.shutdown(d.t))
    r = d.result(rid)
    assert r['status'] == 'failed' and r['reason'] == 'shutdown'


def test_status_carries_request_and_session_only_with_the_skill_interface():
    d = SkillDriver(gb.World(0.1, 2.4, visible=False))
    assert 'skill_session' not in d.fsm.viewer_status(d.t)
    rid = d.request('acquire_caller')
    st = d.fsm.viewer_status(d.t)
    assert st['request_id'] == rid and st['skill_session']['goal_id'] == 'g1'


def test_baseline_trial_without_a_session_has_no_skill_keys():
    d = SkillDriver(gb.World(0.05, 2.4))
    _voice(d)
    d.wake()
    d.run(30)
    s = d.summaries()[-1]
    assert 'goal_id' not in s and 'request_id' not in s
    assert d.results == []


# -- nonsense proposals: whatever the order, nothing unsafe happens --

def test_random_request_sequences_never_move_outside_motion_skills():
    skills = ['localize_caller', 'orient_to_caller', 'acquire_caller', 'approach_caller',
              'sit', 'ask_caller_again', 'cancel']
    rng = random.Random(7)
    for episode in range(60):
        d = SkillDriver(gb.World(rng.uniform(-1.5, 1.5), rng.uniform(1.5, 3.5),
                                 visible=rng.random() > 0.2))
        issued = []
        for _ in range(8):
            if rng.random() < 0.7:
                _voice(d, rng.choice([0.2, 0.9]))
            skill = rng.choice(skills)
            prior = [r for r in d.results if r['status'] == 'succeeded']
            ref = rng.choice([r['request_id'] for r in prior] + ['bogus'])
            args = {'orient_to_caller': {'localization': ref},
                    'approach_caller': {'acquisition': ref},
                    'sit': {'arrival': ref}, 'cancel': {'request_id': ref}}.get(skill, {})
            n = len(d.frames)
            rid = d.request(skill, goal=f'e{episode}', **args)
            issued.append(rid)
            d.run(rng.uniform(0.2, 12))
            for (_, _, c), act in zip(d.frames[n:], d.active[n:]):
                if c.sit:
                    # A sit is only ever sent inside an accepted sit request whose
                    # arrival came from a successful approach.
                    assert act == 'sit'
                    sit_req = [r for r in d.results if r['skill'] == 'sit'
                               and r['status'] != 'rejected']
                    assert sit_req or act == 'sit'
                if c.velocity not in (None, (0.0, 0.0)):
                    assert act == 'approach_caller', act
                if c.rotate_rad is not None:
                    assert act in ('orient_to_caller', 'approach_caller'), act
            for r in d.results:
                if r['skill'] == 'sit' and r['status'] == 'succeeded':
                    arr = [x for x in d.results if x['request_id'] == args.get('arrival')
                           or x['skill'] == 'approach_caller']
                    assert any(x['skill'] == 'approach_caller' and x['status'] == 'succeeded'
                               for x in arr)
        d.run(40)
        for rid in issued:
            assert len([r for r in d.results if r['request_id'] == rid]) == 1, rid
        d.rec('shutdown', d.fsm.shutdown(d.t))


def test_request_id_memory_is_bounded():
    d = SkillDriver(gb.World(0.05, 2.4))
    for i in range(GOAL_ID_MEMORY + 5):
        d.request('ask_caller_again', rid=f'q{i}')
    assert len(d.fsm._seen_request_ids) == GOAL_ID_MEMORY
    assert math.isfinite(d.t)


def test_a_bearing_heard_during_a_cancelled_turn_never_localizes():
    d = SkillDriver(gb.World(1.2, 2.6), rotate_result=False)
    _voice(d)
    loc = d.request('localize_caller')
    d.run_until(loc)
    ori = d.request('orient_to_caller', localization=loc)
    d.run(0.5)
    d.direction(0.8, 0.9)                     # heard mid-turn
    d.run(0.2)
    d.request('cancel', request_id=ori)
    r = d.run_until(d.request('localize_caller'))
    assert r['reason'] == 'no_direction', r


def test_lost_track_bearing_is_not_reused_after_a_failed_turn():
    d = SkillDriver(gb.World(0.05, 3.5), rotate_result=False)
    acq = d.request('acquire_caller')
    d.run_until(acq)
    app = d.request('approach_caller', acquisition=acq)
    d.run(2.0)
    d.world.visible = False
    assert d.run_until(app)['reason'] == 'track_lost'
    t_loss = d.fsm._lost_bearing[1]
    d.fsm._lost_bearing = (0.4, t_loss)          # a distinctive pre-turn bearing
    # Without a turn in between, the reacquisition gates at the lost bearing ...
    probe = d.request('acquire_caller')
    assert d.fsm._gate_center == pytest.approx(0.4)
    d.request('cancel', request_id=probe)
    # ... but after a commanded (here failed) turn it must not.
    d.fsm._lost_bearing = (0.4, t_loss)
    d.direction(1.0, 0.9)
    loc = d.request('localize_caller')
    d.run_until(loc)
    ori = d.request('orient_to_caller', localization=loc)
    assert d.run_until(ori)['reason'] == 'turn_no_result'
    d.request('acquire_caller')
    assert d.fsm._gate_center == 0.0
