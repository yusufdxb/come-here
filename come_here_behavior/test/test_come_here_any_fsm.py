"""Come Here ANY state machine with fake time (pure Python, no ROS).

A kinematic world closes the loop: the robot integrates (vx, vy, yaw_rate)
plus an optional sideways "native avoidance" deflection, publishes odometry
every tick, and person observations come from the geometry. None of this
models the real Unitree avoidance algorithm: these tests prove the controller's
routing, bounds and transitions, not obstacle-avoidance performance.
"""

import math

import pytest

from come_here_behavior.come_here_any_controller import (
    AnyFsmConfig,
    ComeHereAnyFsm,
)
from come_here_behavior.come_here_fsm import ComeHereFsm, FsmConfig, PersonObservation, State

TICK = 0.1
CAMERA_HALF_FOV = 0.9


def obs(bearing=0.0, distance=2.0, conf=0.9, detected=True, bbox=0.5, source=2.0, age=0.1):
    return PersonObservation(bearing, distance, conf, 1.0 if detected else 0.0, bbox, source, age)


def empty(age=0.1):
    return obs(0.0, 0.0, 0.0, False, 0.0, 0.0, age)


class World:
    def __init__(self, bearing, distance):
        self.px = distance * math.cos(bearing)
        self.py = distance * math.sin(bearing)
        self.rx = self.ry = self.heading = 0.0
        self.deflect_vy = 0.0      # sideways drift added while walking forward (a detour)
        self.blocked = False       # translation has no effect (the robot is stuck)
        self.caller_v = (0.0, 0.0)

    def step(self, cmd, dt):
        vx, vy, yaw = cmd
        self.heading += yaw * dt
        if not self.blocked:
            vy_total = vy + (self.deflect_vy if vx > 0.0 else 0.0)
            c, s = math.cos(self.heading), math.sin(self.heading)
            self.rx += (vx * c - vy_total * s) * dt
            self.ry += (vx * s + vy_total * c) * dt
        self.px += self.caller_v[0] * dt
        self.py += self.caller_v[1] * dt

    def distance(self):
        return math.hypot(self.px - self.rx, self.py - self.ry)

    def bearing(self):
        b = math.atan2(self.py - self.ry, self.px - self.rx) - self.heading
        return math.atan2(math.sin(b), math.cos(b))

    def observe(self, conf=0.9):
        b, d = self.bearing(), self.distance()
        if abs(b) > CAMERA_HALF_FOV:
            return empty()
        return obs(bearing=b, distance=d, conf=conf, bbox=min(1.0, 0.45 + 0.25 / d))


def config(**overrides):
    base = dict(skip_turn_to_sound=True, wake_speak_text='', speak_text='',
                sit_hold_until_reset=True, approach_timeout_s=60.0, max_walk_distance_m=10.0)
    base.update(overrides)
    return AnyFsmConfig(**base)


class Sim:
    def __init__(self, world=None, **overrides):
        self.fsm = ComeHereAnyFsm(config(**overrides))
        self.world = world
        self.k = 0
        self.cmd = (0.0, 0.0, 0.0)
        self.velocities = []
        self.summaries = []
        self.transitions = []
        self.logs = []
        self.sits = 0
        self.odom_ok = True

    @property
    def t(self):
        return round(self.k * TICK, 6)

    def _record(self, cmds):
        if cmds.velocity is not None:
            assert len(cmds.velocity) == 3, 'ANY commands are (vx, vy, yaw_rate)'
            self.cmd = cmds.velocity
            self.velocities.append((self.t,) + tuple(cmds.velocity))
        self.sits += int(cmds.sit)
        if cmds.trial_summary is not None:
            self.summaries.append(cmds.trial_summary)
        for line in cmds.log:
            self.logs.append(line)
            if line.startswith('State: '):
                self.transitions.append(line.split('-> ')[1].split(' ')[0])
        return cmds

    def wake(self):
        self.odom()
        return self._record(self.fsm.on_wake('come here', self.t))

    def odom(self):
        w = self.world
        if w is not None and self.odom_ok:
            self._record(self.fsm.on_odom(w.rx, w.ry, w.heading, self.t))

    def run(self, seconds, person='world', every_ticks=2):
        for _ in range(round(seconds / TICK)):
            if self.world is not None:
                self.world.step(self.cmd, TICK)
            self.odom()
            if person is not None and self.k % every_ticks == 0:
                o = self.world.observe() if person == 'world' else (
                    person(self) if callable(person) else person)
                if o is not None:
                    self._record(self.fsm.on_person(o, self.t))
            self._record(self.fsm.tick(self.t))
            self.k += 1

    def run_until(self, pred, max_seconds, **kwargs):
        for _ in range(round(max_seconds / TICK)):
            if pred(self):
                return True
            self.run(TICK, **kwargs)
        return pred(self)

    def phase(self):
        return self.fsm.display_state

    def last_summary(self):
        assert self.summaries, 'no trial summary'
        return self.summaries[-1]


def seated(sim):
    return sim.sits > 0


# -- config validation -----------------------------------------------------

def test_combined_law_needs_its_flag():
    with pytest.raises(ValueError):
        AnyFsmConfig(any_control_law='combined').validate()
    AnyFsmConfig(any_control_law='combined', any_allow_combined=True).validate()


def test_vector_law_needs_lateral_flag():
    with pytest.raises(ValueError):
        AnyFsmConfig(any_control_law='vector').validate()


def test_walk_budget_can_never_be_an_arrival_in_any():
    with pytest.raises(ValueError):
        AnyFsmConfig(walk_budget_arrives=True).validate()


def test_any_requires_the_sit_sequence_with_verified_facing():
    with pytest.raises(ValueError):
        AnyFsmConfig(arrival_mode='stop').validate()


def test_any_fsm_refuses_a_legacy_config():
    with pytest.raises(TypeError):
        ComeHereAnyFsm(FsmConfig())


# -- L: caller centered -> forward intent, arrival ------------------------

def test_centered_caller_walks_forward_only_and_arrives_facing_them():
    sim = Sim(World(0.0, 3.0))
    sim.wake()
    assert sim.run_until(seated, 20.0)
    moving = [v for v in sim.velocities if v[1] or v[2] or v[3]]
    assert moving and all(v[2] == 0.0 for v in moving), 'no lateral motion in split law'
    assert all(not (v[1] and v[3]) for v in moving), 'split law never combines vx and yaw'
    assert any(v[1] > 0.0 for v in moving)
    s = sim.fsm._trial
    assert s is not None and s.final_align_verified is True
    assert s.stop_reason in ('arrived_bbox', 'arrived_distance')


# -- M: caller left / right ------------------------------------------------

@pytest.mark.parametrize('bearing,sign', [(0.5, 1.0), (-0.5, -1.0)])
def test_offset_caller_turns_toward_them_first(bearing, sign):
    sim = Sim(World(bearing, 3.0))
    sim.wake()
    sim.run(0.6)
    first = [v for v in sim.velocities if v[1] or v[3]][0]
    assert first[1] == 0.0 and math.copysign(1.0, first[3]) == sign
    assert sim.run_until(seated, 25.0)


@pytest.mark.parametrize('bearing', [0.3, -0.3])
def test_vector_law_commands_a_body_frame_vector_toward_the_caller(bearing):
    sim = Sim(World(bearing, 3.0), any_control_law='vector', any_allow_lateral=True,
              any_allow_combined=True, approach_align_threshold_rad=0.4,
              approach_realign_threshold_rad=0.5, max_person_bearing_rad=1.0)
    sim.wake()
    sim.run(0.8)
    lateral = [v for v in sim.velocities if v[2] != 0.0]
    assert lateral, 'vector law should command vy for an off-axis caller'
    assert all(math.copysign(1.0, v[2]) == math.copysign(1.0, bearing) for v in lateral)
    assert all(abs(v[2]) <= 0.3 + 1e-9 for v in lateral)
    assert sim.run_until(seated, 25.0)


# -- N: caller moves while approaching -------------------------------------

def test_caller_walking_sideways_is_tracked_continuously():
    world = World(0.0, 3.5)
    world.caller_v = (0.0, 0.15)
    sim = Sim(world)
    sim.wake()
    assert sim.run_until(seated, 30.0)
    assert sim.fsm._trial.final_align_verified is True
    assert abs(sim.fsm._trial.final_align_end_bearing_rad) <= 0.15
    turns = [v for v in sim.velocities if v[3] != 0.0]
    assert turns, 'a moving caller needs re-steering, not one fixed heading'


def test_detour_drift_from_native_avoidance_is_corrected_by_re_steering():
    world = World(0.0, 3.5)
    world.deflect_vy = 0.25          # avoidance pushes the robot sideways while it walks
    sim = Sim(world)
    sim.wake()
    assert sim.run_until(seated, 30.0)
    assert sim.fsm._trial.travel_distance_m > 0.5
    assert abs(world.bearing()) <= 0.15 + 0.05


# -- O / P: occlusion ------------------------------------------------------

def _occluded_between(t0, t1):
    def person(sim):
        if t0 <= sim.t < t1:
            return empty()
        return sim.world.observe()
    return person


def test_brief_occlusion_is_bridged_by_the_caller_estimate():
    sim = Sim(World(0.0, 3.5))
    sim.wake()
    sim.run(2.5)
    assert sim.fsm.state == State.WALK
    t0 = sim.t
    sim.run(0.6, person=_occluded_between(t0, t0 + 0.5))
    assert sim.fsm.state in (State.WALK, State.ALIGN), 'a 0.5 s occlusion must not stop'
    assert sim.run_until(seated, 20.0)
    s = sim.fsm._trial
    assert s.caller_loss_events == 1
    assert s.caller_prediction_used_s > 0.0
    assert s.final_align_verified is True


def test_expired_prediction_stops_and_reacquires():
    sim = Sim(World(0.0, 4.0))
    sim.wake()
    sim.run(2.5)
    t0 = sim.t
    sim.run(2.5, person=_occluded_between(t0, t0 + 10.0))
    assert sim.fsm.state == State.ACQUIRE_PERSON
    assert sim.cmd == (0.0, 0.0, 0.0)
    assert sim.fsm._trial.caller_prediction_expired_events == 1
    assert any('caller_prediction_expired' in line for line in sim.logs)
    sim.run(1.0)                              # visible again: reacquires and continues
    assert sim.fsm.state in (State.WALK, State.ALIGN)


def test_prediction_never_arrives():
    sim = Sim(World(0.0, 1.6))
    sim.wake()
    assert sim.run_until(lambda s: s.fsm.state == State.WALK, 5.0)
    t0 = sim.t
    sim.run(1.0, person=_occluded_between(t0, t0 + 10.0))
    assert sim.fsm.state != State.SIT_AND_IDENTIFY
    assert sim.sits == 0


def test_a_different_person_where_the_caller_was_is_rejected():
    sim = Sim(World(0.0, 3.5))
    sim.wake()
    sim.run(2.0)
    impostor = lambda s: obs(bearing=0.0, distance=s.world.distance() + 2.5,
                             bbox=0.46)
    sim.run(0.6, person=impostor)
    assert sim.fsm._trial.identity_rejections > 0
    assert sim.fsm.state in (State.WALK, State.ALIGN)   # steering on the estimate, not the impostor
    sim.run(2.0, person=impostor)
    assert sim.fsm.state == State.ACQUIRE_PERSON        # estimate expired: stop, never follow them
    assert sim.cmd == (0.0, 0.0, 0.0)


# -- Q: camera stale -------------------------------------------------------

def test_stale_camera_stops_at_once_without_predicting():
    sim = Sim(World(0.0, 3.5))
    sim.wake()
    sim.run(2.5)
    sim.run(0.2, person=lambda s: empty(age=2.0))
    assert sim.cmd == (0.0, 0.0, 0.0)
    assert sim.fsm.state == State.ACQUIRE_PERSON
    s = sim.fsm._trial
    assert s.camera_stale_events == 1 and s.caller_prediction_used_s == 0.0


# -- R / S: bounds ---------------------------------------------------------

def test_no_progress_stops_the_trial():
    world = World(0.0, 3.5)
    world.blocked = True
    sim = Sim(world, any_no_progress_s=4.0)
    sim.wake()
    sim.run(8.0)
    assert sim.fsm.state == State.IDLE
    s = sim.last_summary()
    assert s['stop_reason'] == 'no_progress' and s['success'] is False
    assert s['no_progress_events'] == 1 and sim.sits == 0


def test_travel_budget_stops_and_is_never_an_arrival():
    world = World(0.0, 4.0)
    world.deflect_vy = 0.4
    sim = Sim(world, any_max_travel_m=1.0)
    sim.wake()
    sim.run(10.0)
    s = sim.last_summary()
    assert s['stop_reason'] == 'travel_budget' and s['success'] is False and sim.sits == 0
    assert s['travel_distance_m'] >= 1.0


def test_commanded_walk_budget_aborts_instead_of_arriving():
    sim = Sim(World(0.0, 4.0), max_walk_distance_m=0.8)
    sim.wake()
    sim.run(10.0)
    s = sim.last_summary()
    assert s['stop_reason'] == 'walk_budget' and s['success'] is False and sim.sits == 0


def test_displacement_budget():
    sim = Sim(World(0.0, 4.0), any_max_displacement_m=1.0)
    sim.wake()
    sim.run(10.0)
    assert sim.last_summary()['stop_reason'] == 'displacement_budget'


def test_stale_odometry_stops():
    sim = Sim(World(0.0, 3.5))
    sim.wake()
    sim.run(2.5)
    sim.odom_ok = False
    sim.run(1.0)
    s = sim.last_summary()
    assert s['stop_reason'] == 'odom_stale' and sim.cmd == (0.0, 0.0, 0.0)


def test_odometry_jump_stops():
    sim = Sim(World(0.0, 3.5))
    sim.wake()
    sim.run(2.5)
    sim.world.rx += 1.0
    sim.run(0.3)
    assert sim.last_summary()['stop_reason'] == 'odom_invalid'


def test_trial_timeout():
    sim = Sim(World(0.0, 3.0), any_trial_timeout_s=5.0, search_timeout_s=30.0,
              any_identity_memory_s=30.0)
    sim.wake()
    sim.run(6.0, person=lambda s: empty())
    assert sim.last_summary()['stop_reason'] == 'trial_timeout'


def test_reacquire_timeout_is_reported_as_caller_lost():
    sim = Sim(World(0.0, 4.0), search_timeout_s=3.0)
    sim.wake()
    sim.run(2.5)
    sim.run(6.0, person=lambda s: empty())
    assert sim.last_summary()['stop_reason'] == 'caller_lost'


def test_estop_stops_and_blocks():
    sim = Sim(World(0.0, 3.5))
    sim.wake()
    sim.run(2.5)
    sim._record(sim.fsm.on_estop(True, sim.t))
    assert sim.cmd == (0.0, 0.0, 0.0)
    assert sim.last_summary()['stop_reason'] == 'estop'
    assert sim.fsm.on_wake('come here', sim.t).velocity is None


# -- T / AB: fresh close caller -> arrival -> verified facing -> sit -------

def test_verified_alignment_sits_and_records_success():
    sim = Sim(World(0.0, 1.5))
    sim.wake()
    assert sim.run_until(seated, 15.0)
    sim.run(6.0)
    s = sim.last_summary()
    assert s['success'] is True and s['final_align_verified'] is True
    assert s['mode'] == 'any'
    assert abs(s['final_align_end_bearing_rad']) <= 0.15


# -- U / V / W / X: final facing ignores the caller's own orientation -------

def _arrive_near(bearing, conf, face_present):
    """Caller 0.7 m away at ``bearing``; YOLO confidence and face result differ by pose."""
    world = World(bearing, 0.7)
    sim = Sim(world, bbox_stop_fraction=0.75)
    sim.wake()

    def person(s):
        return s.world.observe(conf=conf)
    for _ in range(200):
        if sim.fsm.display_state == 'LOOK_AT_FACE':
            sim._record(sim.fsm.on_face_result(face_present, sim.t, 0.5 if face_present else None))
        if sim.summaries:
            break
        sim.run(TICK, person=person)
    motion = [v[1:] for v in sim.velocities]
    return sim, motion


def test_final_facing_is_the_same_for_front_side_and_back_facing_callers():
    bearing = math.radians(25.0)
    front, front_cmds = _arrive_near(bearing, conf=0.92, face_present=True)
    side, side_cmds = _arrive_near(bearing, conf=0.74, face_present=False)
    back, back_cmds = _arrive_near(bearing, conf=0.61, face_present=False)
    assert front_cmds == side_cmds == back_cmds, 'caller orientation leaked into robot yaw'
    for sim in (front, side, back):
        s = sim.last_summary()
        assert s['final_align_verified'] is True and s['success'] is True
        assert abs(sim.world.bearing()) <= 0.15 + 0.03
    first_turn = [c for c in front_cmds if c[2] != 0.0][0]
    assert first_turn[2] > 0.0 and first_turn[0] == 0.0 and first_turn[1] == 0.0


def test_final_facing_turns_right_for_a_caller_at_minus_25_deg():
    sim, cmds = _arrive_near(math.radians(-25.0), conf=0.9, face_present=True)
    assert [c for c in cmds if c[2] != 0.0][0][2] < 0.0
    assert sim.last_summary()['final_align_verified'] is True


def test_nothing_about_faces_or_gaze_is_in_the_any_control_law():
    import inspect
    from come_here_behavior import come_here_any_controller as mod
    code = inspect.getsource(mod).split('"""', 2)[2]      # skip the module docstring
    for word in ('face', 'gaze', 'head_pose', 'orientation'):
        assert word not in code.lower().replace('_face_received', ''), word


# -- Y / Z / AA: final alignment failures never sit -------------------------

def _into_final_align(bearing=0.4):
    world = World(bearing, 0.7)
    sim = Sim(world, bbox_stop_fraction=0.75)
    sim.wake()
    assert sim.run_until(lambda s: s.fsm.display_state == 'ALIGN_TO_CALLER', 3.0)
    return sim


def test_stale_bearing_during_final_align_fails_without_sitting():
    sim = _into_final_align()
    sim.run(3.0, person=None)
    s = sim.last_summary()
    assert s['stop_reason'] == 'final_align_failed' and s['success'] is False
    assert s['final_align_verified'] is False and sim.sits == 0
    assert sim.cmd == (0.0, 0.0, 0.0) and sim.fsm.state == State.IDLE


def test_lost_caller_during_final_align_fails_without_sitting():
    sim = _into_final_align()
    sim.run(1.0, person=lambda s: empty())
    s = sim.last_summary()
    assert s['stop_reason'] == 'final_align_failed' and sim.sits == 0


def test_final_align_timeout_fails_instead_of_pretending():
    sim = _into_final_align()
    stuck = lambda s: obs(bearing=0.4, distance=0.7, bbox=0.8)   # bearing never improves
    sim.run(6.0, person=stuck)
    s = sim.last_summary()
    assert s['stop_reason'] == 'final_align_failed' and sim.sits == 0
    assert 'not verified' in s['final_align_failure']


def test_legacy_fsm_still_sits_after_its_final_align_timeout():
    """Regression contract: the hardened facing is ANY-only; legacy is unchanged."""
    fsm = ComeHereFsm(FsmConfig(skip_turn_to_sound=True, arrival_mode='sit_and_identify',
                                final_align_rad=0.15, final_align_timeout_s=1.0,
                                wake_speak_text='', speak_text=''))
    t = 0.0
    fsm.on_wake('come here', t)
    sits = 0
    for k in range(80):
        t = k * TICK
        if k % 2 == 0:
            fsm.on_person(obs(bearing=0.4, distance=0.7, bbox=0.8), t)
        sits += int(fsm.tick(t).sit)
    assert sits == 1


# -- review 2026-09-21 --

def test_identity_memory_must_outlast_the_reacquire_window():
    with pytest.raises(ValueError):
        AnyFsmConfig(any_identity_memory_s=5.0, search_timeout_s=10.0).validate()


def test_a_stranger_after_a_long_occlusion_is_still_rejected():
    sim = Sim(World(0.0, 3.5), search_timeout_s=10.0)
    sim.wake()
    sim.run(2.5)
    sim.run(6.0, person=lambda s: empty())                   # hidden 6 s (> the old 5 s memory)
    impostor = lambda s: obs(bearing=0.0, distance=s.world.distance() + 2.5, bbox=0.46)
    sim.run(2.0, person=impostor)
    assert sim.fsm.state == State.ACQUIRE_PERSON and sim.cmd == (0.0, 0.0, 0.0)
    assert sim.fsm._trial.identity_rejections > 0
    sim.run(4.0, person=impostor)
    assert sim.last_summary()['stop_reason'] == 'caller_lost' and sim.sits == 0


def test_detections_are_rejected_when_odometry_cannot_check_identity():
    sim = Sim(World(0.0, 3.5))
    sim.wake()
    sim.run(2.5)
    sim.run(2.5, person=lambda s: empty())              # lost: stopped in ACQUIRE_PERSON
    assert sim.fsm.state == State.ACQUIRE_PERSON
    sim.odom_ok = False
    sim.run(1.0, person=lambda s: empty())              # odometry now stale
    mark = len(sim.velocities)
    sim.run(2.0)                                        # the caller is visible again
    assert sim.fsm._trial.identity_rejections > 0
    assert sim.fsm.state == State.ACQUIRE_PERSON
    assert all(v[1:] == (0.0, 0.0, 0.0) for v in sim.velocities[mark:])


def _shipped_config(**overrides):
    import pathlib
    import yaml
    cfg = yaml.safe_load((pathlib.Path(__file__).resolve().parents[2] / 'come_here_bringup'
                          / 'config' / 'come_here_any.yaml').read_text())
    params = cfg['behavior_node']['ros__parameters']
    fields = {f for f in AnyFsmConfig.__dataclass_fields__}
    values = {k: v for k, v in params.items() if k in fields}
    values.update(skip_turn_to_sound=True, wake_speak_text='', speak_text='',
                  acquired_speak_text='', relisten_speak_text='')
    values.update(overrides)
    return values


def test_occlusion_keeps_walking_forward_with_the_shipped_config():
    world = World(0.0, 3.5)
    sim = Sim(world)
    sim.fsm = ComeHereAnyFsm(AnyFsmConfig(**_shipped_config(align_by_rotate=False)))
    sim.wake()
    sim.run(3.0)
    assert sim.fsm.state == State.WALK
    t0 = sim.t
    sim.run(1.0, person=_occluded_between(t0, t0 + 0.8))
    during = [v for v in sim.velocities if t0 + 0.35 <= v[0] <= t0 + 0.8]
    assert any(v[1] >= 0.5 for v in during), f'no forward motion while predicting: {during}'
    assert sim.fsm._trial.caller_prediction_used_s > 0.0
