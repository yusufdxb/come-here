"""Stepwise skill interface of ComeHereFsm (fake time, professor_demo parameters).

Invariants under test: one result per request, carrying its goal_id; one active request;
invalid, stale or out-of-order requests are rejected without any motion command; cancel,
timeouts and e-stop end a skill through the existing stop paths; a skill never runs the
baseline's own recovery (relisten, search turns); the baseline path is unaffected.
"""

import dataclasses
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import golden_baseline as gb  # noqa: E402
from come_here_behavior.come_here_fsm import (  # noqa: E402
    GOAL_ID_MEMORY, ComeHereFsm, State,
)


class SkillDriver(gb.Driver):
    """The golden-trace driver, recording skill results instead of refusing them."""

    def __init__(self, world, config=None, **kw):
        super().__init__(world, **kw)
        if config is not None:
            self.fsm = ComeHereFsm(config)
        self.results = []
        self.frames = []

    def rec(self, name, cmds):
        if cmds.velocity is not None:
            self.cmd = cmds.velocity
        if cmds.rotate_rad is not None:
            self.pending_rotate = (cmds.rotate_rad, self.t + 1.0)
        self.frames.append((self.t, name, cmds))
        self.results.extend(cmds.skill_results)
        if cmds.face_request and self.face:
            self.rec('face', self.fsm.on_face_result(True, self.t, 0.5))
        return cmds

    def request(self, gid, skill, v=1, **args):
        req = {'v': v, 'goal_id': gid, 'skill': skill, 'args': args}
        return self.rec('request', self.fsm.on_skill_request(req, self.t))

    def result(self, goal_id):
        found = [r for r in self.results if r['goal_id'] == goal_id]
        assert len(found) == 1, f'{goal_id}: expected one result, got {found}'
        return found[0]

    def summaries(self):
        return [c.trial_summary for _, _, c in self.frames if c.trial_summary]

    def velocities(self):
        return [c.velocity for _, _, c in self.frames if c.velocity is not None]


def _no_motion(cmds):
    return (cmds.velocity is None and cmds.rotate_rad is None and not cmds.sit
            and not cmds.stand and cmds.trial_summary is None)


def _voice(d, conf=0.9):
    d.direction(d.world.bearing(), conf)


# -- turn_to_voice --

def test_turn_succeeds_and_stops_without_approaching():
    d = SkillDriver(gb.World(1.2, 2.4))
    _voice(d)
    d.request('g1', 'turn_to_voice')
    d.run(6)
    r = d.result('g1')
    assert r['status'] == 'succeeded' and r['reason'] == 'turn_complete'
    assert r['skill'] == 'turn_to_voice'
    assert r['data']['turn_result_reason'] == 'reached'
    assert r['data']['turn_rad'] == pytest.approx(1.2, abs=1e-3)
    assert d.fsm.state == State.IDLE
    assert all(v == (0.0, 0.0) for v in d.velocities()), 'a turn skill never walks'
    assert sum(1 for _, _, c in d.frames if c.rotate_rad is not None) == 1
    assert d.summaries()[-1]['goal_id'] == 'g1'


def test_turn_not_needed_when_voice_is_ahead():
    d = SkillDriver(gb.World(0.05, 2.4))
    _voice(d)
    d.request('g1', 'turn_to_voice')
    d.run(2)
    r = d.result('g1')
    assert r['status'] == 'succeeded' and r['reason'] == 'turn_not_needed'
    assert not any(c.rotate_rad is not None for _, _, c in d.frames)
    assert d.fsm.state == State.IDLE


@pytest.mark.parametrize('require_direction', [True, False])
@pytest.mark.parametrize('conf', [None, 0.1])
def test_turn_without_confident_voice_fails_and_never_moves(require_direction, conf):
    cfg = dataclasses.replace(gb.demo_config(), require_direction=require_direction)
    d = SkillDriver(gb.World(0.8, 2.4), config=cfg)
    if conf is not None:
        _voice(d, conf)
    d.request('g1', 'turn_to_voice')
    d.run(12)
    r = d.result('g1')
    assert r['status'] == 'failed' and r['reason'] == 'no_direction'
    assert all(v == (0.0, 0.0) for v in d.velocities())
    assert not any(c.rotate_rad is not None for _, _, c in d.frames)
    assert d.fsm.state == State.IDLE


def test_turn_with_a_stale_bearing_fails():
    d = SkillDriver(gb.World(1.0, 2.4))
    _voice(d)
    d.run(d.fsm.config.direction_max_age_s + 1.0)
    d.request('g1', 'turn_to_voice')
    d.run(12)
    assert d.result('g1')['reason'] == 'no_direction'
    assert not any(c.rotate_rad is not None for _, _, c in d.frames)


def test_turn_without_rotate_result_fails():
    d = SkillDriver(gb.World(1.2, 2.4), rotate_result=False)
    _voice(d)
    d.request('g1', 'turn_to_voice')
    d.run(15)
    r = d.result('g1')
    assert r['status'] == 'failed' and r['reason'] == 'turn_no_result'
    assert d.frames[-1][2].velocity in (None, (0.0, 0.0))
    assert d.fsm.state == State.IDLE


# -- approach_person --

def test_approach_with_stop_arrival_never_sits():
    d = SkillDriver(gb.World(0.05, 2.4))
    d.request('a1', 'approach_person', arrival='stop')
    d.run(30)
    r = d.result('a1')
    assert r['status'] == 'succeeded' and r['reason'].startswith('arrived')
    assert r['data']['arrival'] == 'stop' and 'posture' not in r['data']
    assert not any(c.sit for _, _, c in d.frames)
    assert max(v[0] for v in d.velocities()) <= d.fsm.config.approach_speed + 1e-9
    assert d.velocities()[-1] == (0.0, 0.0)
    assert d.fsm.state == State.IDLE


def test_approach_with_sit_arrival_reports_unverified_posture():
    d = SkillDriver(gb.World(0.05, 2.4))
    d.request('a1', 'approach_person', arrival='sit_and_identify')
    d.run(30)
    r = d.result('a1')
    assert r['status'] == 'succeeded' and r['data']['posture'] == 'unverified'
    assert r['data']['face_present'] is True
    assert sum(1 for _, _, c in d.frames if c.sit) == 1
    # Seated until the operator reset: the next request is refused, nothing moves.
    cmds = d.request('a2', 'approach_person', arrival='stop')
    assert _no_motion(cmds) and d.result('a2')['reason'] == 'not_idle'
    d.rec('reset', d.fsm.on_reset(d.t))
    d.run(4)
    assert d.fsm.state == State.IDLE
    d.request('a3', 'approach_person', arrival='stop')
    assert d.fsm.state == State.ACQUIRE_PERSON


def test_sit_segment_matches_the_baseline_sit_segment():
    """From arrival on, a sit_and_identify skill emits what a baseline trial emits."""
    def segment(d):
        start = next(i for i, (_, _, c) in enumerate(d.frames) if c.sit)
        return [(n, c.velocity, c.sit, c.stand, c.face_request, c.say)
                for _, n, c in d.frames[start - 5:start + 40]]

    base = SkillDriver(gb.World(0.05, 2.4))
    _voice(base)
    base.wake()
    base.run(30)
    skill = SkillDriver(gb.World(0.05, 2.4))
    skill.request('a1', 'approach_person', arrival='sit_and_identify')
    skill.run(30)
    assert segment(skill) == segment(base)


def test_approach_nobody_seen_times_out_without_search_or_relisten():
    d = SkillDriver(gb.World(0.1, 2.4, visible=False))
    d.request('a1', 'approach_person', arrival='stop')
    d.run(d.fsm.config.search_timeout_s + 3)
    r = d.result('a1')
    assert r['status'] == 'failed' and r['reason'] == 'acquire_timeout'
    assert not any(c.rotate_rad is not None for _, _, c in d.frames), 'no search turn'
    assert not any(c.say for _, _, c in d.frames), 'no relisten prompt'
    assert all(v == (0.0, 0.0) for v in d.velocities())


def test_approach_lost_then_reacquired_still_arrives():
    d = SkillDriver(gb.World(0.05, 2.6))
    d.request('a1', 'approach_person', arrival='stop')
    _reach(d, lambda f: f.state == State.WALK)
    d.world.visible = False
    d.run(d.fsm.config.person_stale_timeout_s + 0.5)
    assert d.fsm.state == State.ACQUIRE_PERSON and d.cmd == (0.0, 0.0)
    d.world.visible = True
    d.run(25)
    r = d.result('a1')
    assert r['status'] == 'succeeded' and r['data']['reacquisitions'] >= 1


def test_approach_uses_the_turn_residual_only_right_after_a_turn_skill():
    d = SkillDriver(gb.World(1.2, 2.4))
    _voice(d)
    d.request('t1', 'turn_to_voice')
    d.run(6)
    d.fsm._gate_center = 0.3               # the residual the turn left
    d.request('a1', 'approach_person', arrival='stop')
    assert d.fsm._gate_center == pytest.approx(0.3)
    d.request('c1', 'cancel', goal_id='a1')
    d.request('a2', 'approach_person', arrival='stop')
    assert d.fsm._gate_center == 0.0, 'consumed once, then centered'

    d2 = SkillDriver(gb.World(1.2, 2.4))
    _voice(d2)
    d2.request('t1', 'turn_to_voice')
    d2.run(6)
    d2.fsm._gate_center = 0.3
    d2.rec('estop', d2.fsm.on_estop(True, d2.t))
    d2.rec('estop', d2.fsm.on_estop(False, d2.t))
    d2.request('a1', 'approach_person', arrival='stop')
    assert d2.fsm._gate_center == 0.0, 'an e-stop invalidates the turn residual'


def test_walk_budget_bounds_a_skill_approach():
    d = SkillDriver(gb.World(0.02, 4.5))
    d.request('a1', 'approach_person', arrival='stop')
    d.run(30)
    r = d.result('a1')
    assert r['data']['commanded_walk_distance_m'] <= d.fsm.config.max_walk_distance_m + 0.05


# -- cancel --

def _reach(d, predicate, limit_s=20):
    for _ in range(round(limit_s / gb.TICK)):
        if predicate(d.fsm):
            return
        d.run(gb.TICK)
    raise AssertionError('state not reached')


@pytest.mark.parametrize('where', ['LISTENING', 'TURN_TO_SOUND', 'ACQUIRE_PERSON', 'WALK',
                                   'final_align', 'settle'])
def test_cancel_stops_through_abort_and_leaves_no_estop(where):
    if where in ('LISTENING', 'TURN_TO_SOUND'):
        d = SkillDriver(gb.World(1.2, 2.4))
        d.request('g1', 'turn_to_voice')     # no bearing yet: stays LISTENING
        if where == 'TURN_TO_SOUND':
            _voice(d)
        _reach(d, lambda f: f.state.name == where)
    elif where in ('ACQUIRE_PERSON', 'WALK'):
        d = SkillDriver(gb.World(0.05, 2.6, visible=(where == 'WALK')))
        d.request('g1', 'approach_person', arrival='sit_and_identify')
        _reach(d, lambda f: f.state.name == where)
    else:
        cfg = dataclasses.replace(gb.demo_config(), final_align_rad=0.005)
        d = SkillDriver(gb.World(0.2 if where == 'final_align' else 0.0, 2.4),
                        config=cfg if where == 'final_align' else None)
        d.request('g1', 'approach_person', arrival='sit_and_identify')
        _reach(d, lambda f: f.state == State.SIT_AND_IDENTIFY and f._sit_phase == where)
    n = len(d.frames)
    cmds = d.request('c1', 'cancel', goal_id='g1')
    assert cmds.velocity == (0.0, 0.0)
    assert not cmds.sit and cmds.rotate_rad is None
    assert d.fsm.state == State.IDLE and not d.fsm.estopped
    assert d.result('g1')['status'] == 'cancelled'
    assert d.result('g1')['data']['cancel_goal_id'] == 'c1'
    assert d.result('c1')['status'] == 'succeeded'
    d.run(3)
    after = [c for _, _, c in d.frames[n:]]
    assert all(c.velocity in (None, (0.0, 0.0)) for c in after)
    assert not any(c.sit or c.stand or c.rotate_rad is not None for c in after)
    d.request('g2', 'approach_person', arrival='stop')
    assert not [r for r in d.results if r['goal_id'] == 'g2'], 'accepted, still running'
    assert d.fsm.state == State.ACQUIRE_PERSON


@pytest.mark.parametrize('phase', ['sit', 'look', 'done'])
def test_cancel_during_posture_phases_is_refused(phase):
    d = SkillDriver(gb.World(0.0, 2.4), face=False)
    d.request('g1', 'approach_person', arrival='sit_and_identify')
    _reach(d, lambda f: f.state == State.SIT_AND_IDENTIFY and f._sit_phase == phase, 40)
    active = d.fsm.active_goal_id
    cmds = d.request('c1', 'cancel', goal_id='g1')
    assert _no_motion(cmds)
    assert d.result('c1')['reason'] == ('posture_phase' if active else 'not_active')


def test_cancel_of_another_goal_is_rejected_and_the_active_skill_continues():
    d = SkillDriver(gb.World(0.05, 2.6))
    d.request('g2', 'approach_person', arrival='stop')
    _reach(d, lambda f: f.state == State.WALK)
    cmds = d.request('c1', 'cancel', goal_id='g1')      # a stale cancel for an old goal
    assert _no_motion(cmds) and d.result('c1')['reason'] == 'not_active'
    assert d.fsm.state == State.WALK and d.fsm.active_goal_id == 'g2'
    d.run(30)
    assert d.result('g2')['status'] == 'succeeded'


def test_cancel_with_nothing_active_is_rejected():
    d = SkillDriver(gb.World(0.05, 2.6))
    cmds = d.request('c1', 'cancel', goal_id='g1')
    assert _no_motion(cmds) and d.result('c1')['reason'] == 'not_active'


# -- rejections: never move --

@pytest.mark.parametrize('req,reason', [
    ('not a dict', 'malformed'),
    ({'v': 2, 'goal_id': 'x', 'skill': 'turn_to_voice', 'args': {}}, 'malformed'),
    ({'v': 1, 'skill': 'turn_to_voice', 'args': {}}, 'malformed'),
    ({'v': 1, 'goal_id': '', 'skill': 'turn_to_voice', 'args': {}}, 'malformed'),
    ({'v': 1, 'goal_id': 'x' * 65, 'skill': 'turn_to_voice', 'args': {}}, 'malformed'),
    ({'v': 1, 'goal_id': 7, 'skill': 'turn_to_voice', 'args': {}}, 'malformed'),
    ({'v': 1, 'goal_id': 'x', 'skill': 'turn_to_voice', 'args': []}, 'malformed'),
    ({'v': 1, 'goal_id': 'x', 'skill': 'walk_forward', 'args': {}}, 'unknown_skill'),
    ({'v': 1, 'goal_id': 'x', 'skill': 'turn_to_voice', 'args': {'angle': 1.0}}, 'bad_args'),
    ({'v': 1, 'goal_id': 'x', 'skill': 'approach_person', 'args': {}}, 'bad_args'),
    ({'v': 1, 'goal_id': 'x', 'skill': 'approach_person',
      'args': {'arrival': 'run'}}, 'bad_args'),
    ({'v': 1, 'goal_id': 'x', 'skill': 'approach_person',
      'args': {'arrival': 'stop', 'speed': 1.0}}, 'bad_args'),
    ({'v': 1, 'goal_id': 'x', 'skill': 'cancel', 'args': {}}, 'bad_args'),
])
def test_invalid_requests_are_rejected_without_motion(req, reason):
    fsm = ComeHereFsm(gb.demo_config())
    cmds = fsm.on_skill_request(req, 1.0)
    assert _no_motion(cmds)
    assert [r['status'] for r in cmds.skill_results] == ['rejected']
    assert cmds.skill_results[0]['reason'] == reason
    assert fsm.state == State.IDLE and not fsm.trial_active


def test_only_one_active_request():
    d = SkillDriver(gb.World(1.2, 2.4))
    _voice(d)
    d.request('t1', 'turn_to_voice')
    cmds = d.request('a1', 'approach_person', arrival='stop')
    assert _no_motion(cmds) and d.result('a1')['reason'] == 'not_idle'
    assert d.fsm.active_goal_id == 't1'
    d.run(6)
    assert d.result('t1')['status'] == 'succeeded'


def test_requests_are_refused_during_a_baseline_trial():
    d = SkillDriver(gb.World(0.05, 2.4))
    _voice(d)
    d.wake()
    d.run(1)
    cmds = d.request('a1', 'approach_person', arrival='stop')
    assert _no_motion(cmds) and d.result('a1')['reason'] == 'not_idle'
    cmds = d.request('c1', 'cancel', goal_id='a1')
    assert _no_motion(cmds) and d.result('c1')['reason'] == 'not_active'
    assert d.fsm.trial_active


def test_requests_are_refused_while_estopped():
    d = SkillDriver(gb.World(0.05, 2.4))
    d.rec('estop', d.fsm.on_estop(True, d.t))
    cmds = d.request('a1', 'approach_person', arrival='stop')
    assert _no_motion(cmds) and d.result('a1')['reason'] == 'estopped'


def test_a_goal_id_is_never_reused():
    d = SkillDriver(gb.World(0.05, 2.4))
    d.request('g1', 'turn_to_voice')
    d.run(12)                                   # no bearing: fails no_direction
    assert d.result('g1')['status'] == 'failed'
    cmds = d.request('g1', 'approach_person', arrival='stop')
    assert _no_motion(cmds)
    rejected = [r for r in d.results if r['goal_id'] == 'g1' and r['status'] == 'rejected']
    assert [r['reason'] for r in rejected] == ['duplicate_goal_id']
    assert d.fsm.state == State.IDLE


def test_goal_id_memory_is_bounded():
    fsm = ComeHereFsm(gb.demo_config())
    for i in range(GOAL_ID_MEMORY + 10):
        fsm.on_skill_request({'v': 1, 'goal_id': f'c{i}', 'skill': 'cancel',
                              'args': {'goal_id': 'none'}}, 1.0)
    assert len(fsm._seen_goal_ids) == GOAL_ID_MEMORY


# -- interaction with the baseline, e-stop and shutdown --

def test_a_wake_during_a_skill_is_ignored():
    d = SkillDriver(gb.World(0.05, 2.6))
    d.request('a1', 'approach_person', arrival='stop')
    d.run(1)
    trial_before = d.fsm._trial
    d.wake()
    assert d.fsm._trial is trial_before and d.fsm.active_goal_id == 'a1'


def test_estop_during_a_skill_reports_cancelled_estop():
    d = SkillDriver(gb.World(0.05, 2.6))
    d.request('a1', 'approach_person', arrival='stop')
    _reach(d, lambda f: f.state == State.WALK)
    cmds = d.rec('estop', d.fsm.on_estop(True, d.t))
    assert cmds.velocity == (0.0, 0.0)
    r = d.result('a1')
    assert r['status'] == 'cancelled' and r['reason'] == 'estop'
    assert d.fsm.state == State.IDLE and d.fsm.estopped


def test_shutdown_during_a_skill_reports_failed():
    d = SkillDriver(gb.World(0.05, 2.6))
    d.request('a1', 'approach_person', arrival='stop')
    _reach(d, lambda f: f.state == State.WALK)
    cmds = d.rec('shutdown', d.fsm.shutdown(d.t))
    assert cmds.velocity == (0.0, 0.0)
    r = d.result('a1')
    assert r['status'] == 'failed' and r['reason'] == 'shutdown'


def test_baseline_trial_after_a_skill_has_no_skill_keys():
    d = SkillDriver(gb.World(0.05, 2.4))
    d.request('a1', 'approach_person', arrival='stop')
    d.run(30)
    _voice(d)
    d.wake()
    d.run(30)
    first, last = d.summaries()[0], d.summaries()[-1]
    assert first['goal_id'] == 'a1' and first['skill'] == 'approach_person'
    assert 'goal_id' not in last and 'skill' not in last
    assert set(last) == set(first) - {'goal_id', 'skill'}
    assert len(d.results) == 1, 'a baseline trial emits no skill result'


def test_status_carries_goal_id_only_while_a_skill_is_active():
    d = SkillDriver(gb.World(0.05, 2.4))
    assert 'goal_id' not in d.fsm.viewer_status(d.t)
    d.request('a1', 'approach_person', arrival='stop')
    status = d.fsm.viewer_status(d.t)
    assert status['goal_id'] == 'a1' and status['skill'] == 'approach_person'
    d.run(30)
    assert 'goal_id' not in d.fsm.viewer_status(d.t)


def test_every_accepted_request_ends_with_exactly_one_terminal_result():
    d = SkillDriver(gb.World(1.2, 2.4))
    _voice(d)
    d.request('t1', 'turn_to_voice')
    d.run(6)
    d.request('a1', 'approach_person', arrival='stop')
    d.run(30)
    d.request('t2', 'turn_to_voice')
    d.run(12)
    ids = [r['goal_id'] for r in d.results]
    assert sorted(ids) == ['a1', 't1', 't2'] and len(set(ids)) == 3
    assert all(r['status'] in ('succeeded', 'failed', 'cancelled') for r in d.results)
    assert all(math.isfinite(v[0]) for v in d.velocities())
