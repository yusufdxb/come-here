"""State-machine tests for ComeHereFsm with fake time (pure Python, no ROS).

Every test drives the FSM only through its inputs (wake, person observations,
e-stop, ticks) and checks its outputs (velocity commands, speech, state
transitions, trial summaries). A tiny kinematic world model closes the loop
for the end-to-end scenarios: the robot integrates the last velocity command
and observations are generated from the resulting geometry.
"""

import math
import random

import pytest

from come_here_behavior.come_here_fsm import (
    ARRIVAL_SIT_AND_IDENTIFY,
    ComeHereFsm,
    FsmConfig,
    PersonObservation,
    State,
)

TICK = 0.1


def obs(bearing=0.0, distance=2.0, conf=0.9, detected=True, bbox=0.5,
        source=2.0, age=0.1):
    return PersonObservation(
        bearing, distance, conf, 1.0 if detected else 0.0, bbox, source, age
    )


MISS = obs(bearing=0.0, distance=0.0, conf=0.0, detected=False, bbox=0.0, source=0.0)


class World:
    """Robot pose integrates commands; observations come from the geometry.

    bbox_h_frac grows as the person gets closer (0.55 at 2.5 m, 0.75 at about
    0.83 m). The distance field carries the ~+1 m LiDAR bias seen while
    walking on 2026-04-24, so only the bbox trigger can stop the robot.
    """

    def __init__(self, bearing, distance):
        self.px = distance * math.cos(bearing)
        self.py = distance * math.sin(bearing)
        self.rx = self.ry = self.heading = 0.0

    def step(self, vx, yaw_rate, dt):
        self.heading += yaw_rate * dt
        self.rx += vx * dt * math.cos(self.heading)
        self.ry += vx * dt * math.sin(self.heading)

    def distance(self):
        return math.hypot(self.px - self.rx, self.py - self.ry)

    def observe(self, noise=0.0):
        dx, dy = self.px - self.rx, self.py - self.ry
        b = math.atan2(dy, dx) - self.heading
        b = math.atan2(math.sin(b), math.cos(b)) + noise
        d = self.distance()
        return obs(bearing=b, distance=d + 1.0, bbox=min(1.0, 0.45 + 0.25 / d))


class Sim:
    """Drives the FSM on a 10 Hz fake clock and records everything it emits."""

    def __init__(self, **overrides):
        self.fsm = ComeHereFsm(FsmConfig(**overrides))
        self.k = 0
        self.cmd = (0.0, 0.0)
        self.velocities = []
        self.says = []
        self.summaries = []
        self.transitions = []
        self.rotates = []
        self.sits = self.stands = self.face_requests = 0

    @property
    def t(self):
        return round(self.k * TICK, 6)

    def _record(self, cmds):
        if cmds.velocity is not None:
            self.cmd = cmds.velocity
            self.velocities.append((self.t,) + cmds.velocity)
        if cmds.say:
            self.says.append(cmds.say)
        if cmds.rotate_rad is not None:
            self.rotates.append(cmds.rotate_rad)
        self.sits += int(cmds.sit)
        self.stands += int(cmds.stand)
        self.face_requests += int(cmds.face_request)
        if cmds.trial_summary is not None:
            self.summaries.append(cmds.trial_summary)
        for line in cmds.log:
            if line.startswith('State: '):
                self.transitions.append(line.split('-> ')[1].split(' ')[0])
        return cmds

    def wake(self):
        return self._record(self.fsm.on_wake('come here', self.t))

    def person(self, o):
        return self._record(self.fsm.on_person(o, self.t))

    def estop(self, engaged):
        return self._record(self.fsm.on_estop(engaged, self.t))

    def run(self, seconds, person=None, every_ticks=2, world=None):
        """Advance ``seconds``; deliver ``person`` (obs or callable) every N ticks."""
        for _ in range(round(seconds / TICK)):
            if world is not None:
                world.step(self.cmd[0], self.cmd[1], TICK)
            if person is not None and self.k % every_ticks == 0:
                o = person(self) if callable(person) else person
                if o is not None:
                    self.person(o)
            self._record(self.fsm.tick(self.t))
            self.k += 1

    def run_until(self, state, max_seconds, **kwargs):
        for _ in range(round(max_seconds / TICK)):
            if self.fsm.state == state:
                return True
            self.run(TICK, **kwargs)
        return self.fsm.state == state

    def motion_commands(self, since=0.0):
        return [(t, vx, w) for t, vx, w in self.velocities
                if t >= since and (vx != 0.0 or w != 0.0)]


def assert_single_axis(sim):
    combined = [v for v in sim.velocities if v[1] != 0.0 and v[2] != 0.0]
    assert combined == [], f'combined forward+yaw commands emitted: {combined}'


def walking_sim(**overrides):
    """A sim that has acquired a centered person and is in WALK."""
    sim = Sim(**overrides)
    sim.wake()
    assert sim.run_until(State.WALK, 3.0, person=obs(bearing=0.0))
    return sim


# -- nominal end-to-end --

def test_nominal_centered_caller_walks_straight_and_stops():
    sim = Sim()
    world = World(bearing=0.03, distance=2.5)
    wake_cmds = sim.wake()
    assert wake_cmds.velocity == (0.0, 0.0)
    assert sim.says == ['I am coming']
    sim.run(12.0, person=lambda s: world.observe(), every_ticks=3, world=world)

    assert sim.transitions == ['ACQUIRE_PERSON', 'WALK', 'ARRIVED', 'IDLE']
    assert_single_axis(sim)
    assert sim.velocities[-1][1:] == (0.0, 0.0)
    assert sim.says == ['I am coming', 'I am here']
    assert 0.6 <= world.distance() <= 1.1
    [summary] = sim.summaries
    assert summary['stop_reason'] == 'arrived_bbox'
    assert summary['success'] is True
    assert (summary['align_phases'], summary['walk_phases']) == (0, 1)
    assert summary['combined_commands'] == 0
    assert summary['lost_events'] == summary['stale_events'] == 0
    assert summary['first_detection_latency_s'] >= 0.0
    assert summary['acquire_latency_s'] >= summary['first_detection_latency_s']


def test_nominal_misaligned_caller_aligns_walks_and_stops():
    sim = Sim()
    world = World(bearing=0.5, distance=2.5)
    sim.wake()
    sim.run(20.0, person=lambda s: world.observe(), every_ticks=3, world=world)

    assert sim.transitions[:3] == ['ACQUIRE_PERSON', 'ALIGN', 'WALK']
    assert sim.transitions[-2:] == ['ARRIVED', 'IDLE']
    assert_single_axis(sim)
    assert world.distance() >= 0.6
    [summary] = sim.summaries
    assert summary['success'] is True
    assert summary['align_phases'] >= 1 and summary['walk_phases'] >= 1


def test_jittery_bearing_still_arrives_with_single_axis_commands():
    rng = random.Random(7)
    sim = Sim()
    world = World(bearing=-0.3, distance=2.5)
    sim.wake()
    sim.run(30.0, person=lambda s: world.observe(noise=rng.uniform(-0.2, 0.2)),
            every_ticks=3, world=world)
    assert_single_axis(sim)
    assert world.distance() >= 0.6
    assert sim.summaries[-1]['success'] is True


def test_professor_mode_never_sits_or_runs_face_detection():
    sim = Sim()
    world = World(bearing=0.0, distance=2.0)
    sim.wake()
    sim.run(12.0, person=lambda s: world.observe(), every_ticks=3, world=world)
    assert sim.summaries[-1]['success'] is True
    assert (sim.sits, sim.stands, sim.face_requests, sim.rotates) == (0, 0, 0, [])


# -- acquisition --

def test_already_aligned_goes_straight_to_walk():
    sim = Sim()
    sim.wake()
    sim.run(1.0, person=obs(bearing=0.05))
    assert sim.transitions == ['ACQUIRE_PERSON', 'WALK']
    assert sim.motion_commands()[0][1:] == (0.6, 0.0)


@pytest.mark.parametrize('bearing,expected_yaw', [(0.4, 0.6), (-0.4, -0.6)])
def test_misaligned_first_motion_is_yaw_only_toward_person(bearing, expected_yaw):
    sim = Sim()
    sim.wake()
    sim.run(1.0, person=obs(bearing=bearing))
    assert sim.transitions == ['ACQUIRE_PERSON', 'ALIGN']
    assert sim.motion_commands()[0][1:] == (0.0, expected_yaw)


def test_single_detection_frame_does_not_start_motion():
    sim = Sim()
    sim.wake()
    sim.person(obs())
    sim.run(3.0)
    assert sim.motion_commands() == []
    assert sim.fsm.state == State.ACQUIRE_PERSON


def test_detections_before_wake_do_not_count():
    sim = Sim()
    for _ in range(5):
        sim.person(obs())
    sim.wake()
    sim.run(0.5)
    assert sim.motion_commands() == []
    sim.person(obs())
    sim.run(0.1)
    assert sim.motion_commands() == []


def test_low_confidence_detection_never_acquires():
    sim = Sim()
    sim.wake()
    sim.run(5.0, person=obs(conf=0.3))
    assert sim.motion_commands() == []


def test_search_timeout_returns_to_idle_without_motion():
    sim = Sim()
    sim.wake()
    sim.run(10.5)
    assert sim.fsm.state == State.IDLE
    assert sim.motion_commands() == []
    assert sim.velocities[-1][1:] == (0.0, 0.0)
    assert sim.summaries[-1]['stop_reason'] == 'acquire_timeout'
    assert sim.summaries[-1]['success'] is False


def test_wake_ignored_while_trial_active():
    sim = walking_sim()
    before = (len(sim.velocities), list(sim.says))
    sim.wake()
    assert (len(sim.velocities), sim.says) == before
    assert sim.fsm.state == State.WALK


# -- ALIGN / WALK hysteresis --

def test_realign_after_min_walk_when_bearing_exceeds_threshold():
    sim = walking_sim()
    walk_start = sim.t
    sim.run(1.2, person=obs(bearing=0.02))
    sim.run(2.0, person=obs(bearing=0.5))
    assert 'ALIGN' in sim.transitions
    t_align = next(t for t, vx, w in sim.velocities if t > walk_start and vx == 0.0 and w != 0.0)
    assert t_align - walk_start >= 1.5
    assert sim.velocities[-1][1:] == (0.0, 0.6)


def test_no_realign_before_min_walk_s():
    sim = walking_sim()
    sim.run(1.0, person=obs(bearing=0.6))
    assert sim.fsm.state == State.WALK
    assert all(vx == 0.6 and w == 0.0 for _, vx, w in sim.motion_commands())


def test_bearing_between_deadband_and_realign_never_flaps():
    sim = walking_sim()
    sim.run(6.0, person=obs(bearing=0.22))
    assert sim.transitions == ['ACQUIRE_PERSON', 'WALK']

    sim = Sim()
    sim.wake()
    sim.run(6.0, person=obs(bearing=0.22))
    assert sim.transitions == ['ACQUIRE_PERSON', 'ALIGN']


def test_single_jitter_frame_does_not_trigger_realign():
    sim = walking_sim()
    sim.run(1.6, person=obs(bearing=0.0))
    sim.person(obs(bearing=0.6))
    sim.run(0.2, person=obs(bearing=0.0))
    assert sim.fsm.state == State.WALK


# -- lost person --

def test_lost_person_during_walk_stops_immediately_and_searches():
    sim = walking_sim()
    sim.run(0.6, person=obs())
    first_miss = sim.t
    sim.run(1.0, person=MISS)
    stops = [t for t, vx, w in sim.velocities if t >= first_miss and (vx, w) == (0.0, 0.0)]
    assert stops and stops[0] - first_miss <= 0.4
    assert sim.fsm.state == State.ACQUIRE_PERSON
    assert sim.motion_commands(since=stops[0]) == []


def test_motion_resumes_only_after_n_fresh_detections():
    sim = walking_sim()
    sim.run(1.0, person=MISS)
    assert sim.fsm.state == State.ACQUIRE_PERSON
    resume_from = sim.t
    sim.person(obs())
    sim.run(0.3)
    assert sim.motion_commands(since=resume_from) == []
    sim.person(obs())
    sim.run(0.1)
    assert sim.motion_commands(since=resume_from) != []
    sim.run(0.1, person=obs())
    assert sim.fsm.state == State.WALK


def test_stale_perception_stops_when_messages_stop():
    sim = walking_sim(person_stale_timeout_s=1.5)
    last_obs = sim.t
    sim.run(3.0)  # perception goes silent
    assert sim.fsm.state == State.ACQUIRE_PERSON
    last_motion = sim.motion_commands()[-1][0]
    assert last_motion - last_obs <= 1.6
    assert sim.velocities[-1][1:] == (0.0, 0.0)


def test_reacquire_timeout_returns_to_idle():
    sim = walking_sim()
    sim.run(1.0, person=MISS)
    sim.run(10.5, person=MISS)
    assert sim.fsm.state == State.IDLE
    summary = sim.summaries[-1]
    assert summary['stop_reason'] == 'reacquire_timeout'
    assert summary['lost_events'] == 1
    assert summary['success'] is False


# -- invalid perception --

@pytest.mark.parametrize('bad', [
    obs(bearing=math.nan),
    obs(distance=math.inf),
    obs(bearing=5.0),
    obs(distance=-1.0),
    obs(conf=1.5),
    obs(bbox=3.0),
    PersonObservation(0.0, 2.0, 0.9, 0.5, 0.5, 2.0, 0.1),
    None,
], ids=['nan_bearing', 'inf_distance', 'bearing_out_of_range', 'negative_distance',
        'confidence_above_one', 'bbox_above_one', 'detected_not_boolean', 'malformed'])
def test_invalid_perception_never_produces_motion(bad):
    sim = Sim()
    sim.wake()
    sim.run(4.0, person=lambda s: bad)
    assert sim.motion_commands() == []
    assert sim.fsm.state == State.ACQUIRE_PERSON


def test_invalid_perception_during_walk_stops_like_a_miss():
    sim = walking_sim()
    first_bad = sim.t
    sim.run(1.0, person=obs(bearing=math.nan), every_ticks=1)
    assert sim.fsm.state == State.ACQUIRE_PERSON
    stops = [t for t, vx, w in sim.velocities if t >= first_bad and (vx, w) == (0.0, 0.0)]
    assert stops and stops[0] - first_bad <= 0.4
    assert sim.motion_commands(since=stops[0]) == []
    sim.run(10.5)
    assert sim.summaries[-1]['invalid_observations'] >= 1


@pytest.mark.parametrize('data,valid', [
    ([0.1, 2.0, 0.9, 1.0], True),
    ([0.1, 2.0, 0.9, 1.0, 0.5, 2.0, 0.1], True),
    ([0.1, 2.0, 0.9], False),
    ([0.1, 2.0, 0.9, 1.0, 0.5, 2.0, 0.1, 9.0], False),
])
def test_person_array_parsing(data, valid):
    assert (PersonObservation.from_array(data) is not None) == valid


# -- stopping --

def test_distance_trigger_arrives_when_bbox_is_small():
    sim = walking_sim()
    sim.run(0.5, person=obs(distance=0.7, bbox=0.4))
    assert sim.fsm.state == State.ARRIVED
    assert sim.velocities[-1][1:] == (0.0, 0.0)
    sim.run(2.5)
    assert sim.summaries[-1]['stop_reason'] == 'arrived_distance'


def test_close_person_at_acquisition_arrives_without_moving():
    sim = Sim()
    sim.wake()
    sim.run(1.0, person=obs(bbox=0.9))
    assert sim.motion_commands() == []
    assert sim.fsm.state == State.ARRIVED


def test_walk_budget_stops_even_if_perception_never_says_close():
    sim = walking_sim(max_walk_distance_m=2.2)
    sim.run(10.0, person=obs(bbox=0.5, distance=3.0))
    assert sim.fsm.state == State.IDLE
    summary = sim.summaries[-1]
    assert summary['stop_reason'] == 'walk_budget'
    assert 2.2 <= summary['commanded_walk_distance_m'] <= 2.3
    assert summary['success'] is False
    assert sim.velocities[-1][1:] == (0.0, 0.0)


def test_approach_timeout_stops_a_never_converging_align():
    sim = Sim(approach_timeout_s=5.0)
    sim.wake()
    sim.run(8.0, person=obs(bearing=0.5))  # the static bearing never converges
    assert sim.fsm.state == State.IDLE
    assert sim.summaries[-1]['stop_reason'] == 'approach_timeout'
    assert sim.velocities[-1][1:] == (0.0, 0.0)


# -- e-stop and shutdown --

def test_estop_during_walk_stops_and_blocks_future_motion():
    sim = walking_sim()
    cmds = sim.estop(True)
    assert cmds.velocity == (0.0, 0.0)
    assert sim.fsm.state == State.IDLE
    assert sim.summaries[-1]['stop_reason'] == 'estop'
    assert sim.summaries[-1]['success'] is False

    t0 = sim.t
    sim.wake()
    sim.run(3.0, person=obs())
    assert sim.motion_commands(since=t0) == []
    assert sim.fsm.state == State.IDLE


def test_estop_release_does_not_resume_until_a_new_wake():
    sim = walking_sim()
    sim.estop(True)
    sim.estop(False)
    t0 = sim.t
    sim.run(2.0, person=obs())
    assert sim.motion_commands(since=t0) == []
    wake_cmds = sim.wake()
    assert wake_cmds.velocity == (0.0, 0.0)
    sim.run(1.0, person=obs())
    assert sim.fsm.state == State.WALK


def test_estop_after_arrival_keeps_the_arrival_result():
    sim = walking_sim()
    sim.run(0.5, person=obs(bbox=0.8))
    assert sim.fsm.state == State.ARRIVED
    sim.estop(True)
    summary = sim.summaries[-1]
    assert summary['stop_reason'] == 'arrived_bbox'
    assert summary['estop_after_stop'] is True


def test_shutdown_during_walk_stops_and_reports():
    sim = walking_sim()
    cmds = sim.fsm.shutdown(sim.t)
    assert cmds.velocity == (0.0, 0.0)
    assert cmds.trial_summary['stop_reason'] == 'shutdown'
    assert sim.fsm.state == State.IDLE


# -- optional behaviors --

def test_sit_and_identify_mode_runs_full_sequence():
    sim = walking_sim(arrival_mode=ARRIVAL_SIT_AND_IDENTIFY)
    sim.run(0.3, person=obs(bbox=0.8))
    assert sim.fsm.state == State.SIT_AND_IDENTIFY
    assert sim.sits == 0                # stop first, sit only after pre_sit_settle_s
    sim.run(1.0)
    assert sim.sits == 1
    sim.run(12.0)
    assert (sim.face_requests, sim.stands) == (1, 1)
    assert sim.says[-1] == 'I am here'
    assert sim.fsm.state == State.IDLE
    assert sim.summaries[-1]['success'] is True


def test_turn_to_sound_path_rotates_then_acquires():
    sim = Sim(skip_turn_to_sound=False)
    sim.fsm.on_direction(0.8, 0.9, sim.t)
    sim.wake()
    sim.run(0.3)
    assert sim.transitions == ['LISTENING', 'TURN_TO_SOUND']
    sim._record(sim.fsm.on_rotate_result(0.8, 0.75, 'reached', sim.t))
    sim.run(1.0)
    assert sim.transitions[:3] == ['LISTENING', 'TURN_TO_SOUND', 'ACQUIRE_PERSON']
    assert sim.rotates == [0.8]


def test_direction_published_just_after_the_wake_still_turns():
    sim = Sim(skip_turn_to_sound=False)
    sim.wake()
    sim.run(0.3)
    sim.fsm.on_direction(-1.2, 0.8, sim.t)
    sim.run(0.3)
    assert sim.transitions[:2] == ['LISTENING', 'TURN_TO_SOUND']
    assert sim.rotates == [-1.2]
    assert sim.fsm.status  # trial still open


def test_a_stale_direction_from_an_earlier_utterance_is_ignored():
    sim = Sim(skip_turn_to_sound=False, direction_max_age_s=3.0)
    sim.fsm.on_direction(1.0, 0.9, sim.t)
    sim.run(5.0)                       # quiet for longer than direction_max_age_s
    sim.wake()
    sim.run(2.0)
    assert sim.transitions == ['LISTENING', 'ACQUIRE_PERSON']
    assert sim.rotates == []


def test_a_caller_already_ahead_is_not_turned_toward():
    sim = Sim(skip_turn_to_sound=False, turn_min_rad=0.2)
    sim.fsm.on_direction(0.1, 0.9, sim.t)
    sim.wake()
    sim.run(0.3)
    assert sim.transitions == ['LISTENING', 'ACQUIRE_PERSON']
    assert sim.rotates == []


def test_the_turn_is_recorded_in_the_trial_summary():
    sim = Sim(skip_turn_to_sound=False)
    sim.fsm.on_direction(0.8, 0.9, sim.t)
    sim.wake()
    sim.run(12.0)                      # no person: search timeout ends the trial
    assert sim.summaries[-1]['turn_rad'] == 0.8
    assert sim.summaries[-1]['turn_confidence'] == 0.9


def test_turn_to_sound_falls_back_without_confident_direction():
    sim = Sim(skip_turn_to_sound=False)
    sim.wake()
    sim.run(2.0)
    assert sim.transitions == ['LISTENING', 'ACQUIRE_PERSON']
    assert sim.rotates == []


def test_empty_speech_texts_disable_tts():
    sim = walking_sim(wake_speak_text='', speak_text='')
    sim.run(0.3, person=obs(bbox=0.8))
    assert sim.says == []


# -- configuration --

@pytest.mark.parametrize('overrides', [
    dict(approach_align_threshold_rad=0.3, approach_realign_threshold_rad=0.3),
    dict(bbox_stop_fraction=1.5),
    dict(arrival_mode='dance'),
    dict(bearing_ema_alpha=0.0),
    dict(approach_speed=-0.6),
    dict(search_min_consecutive_detections=0),
    dict(person_stale_timeout_s=math.nan),
])
def test_invalid_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        ComeHereFsm(FsmConfig(**overrides))


# -- DOA-gated acquisition, speech and the seated finish (class demo 2026-09-14) --

def turning_sim(target=0.8, confidence=0.9, **overrides):
    """Wake with a confident bearing; the rotate command has been sent."""
    sim = Sim(skip_turn_to_sound=False, **overrides)
    sim.fsm.on_direction(target, confidence, sim.t)
    sim.wake()
    sim.run(0.3)
    return sim


def test_no_person_is_acquired_while_the_robot_is_still_turning():
    sim = turning_sim()
    assert sim.fsm.state == State.TURN_TO_SOUND and sim.rotates == [0.8]
    sim.run(3.0, person=obs(bearing=-0.5), every_ticks=1)   # a bystander swept past
    assert sim.fsm.state == State.TURN_TO_SOUND
    assert sim.motion_commands() == []
    assert sim.rotates == [0.8]                             # exactly one turn
    assert sim.fsm.tick(sim.t).gate == (0.0, 0.0)           # perception selects nobody


def test_gate_is_centered_where_the_voice_is_after_the_turn():
    sim = turning_sim(target=0.8)
    sim._record(sim.fsm.on_rotate_result(0.8, 0.6, 'reached', sim.t))
    sim.run(0.5)
    assert sim.fsm.state == State.TURN_TO_SOUND             # turn_settle_s
    sim.run(0.4)
    assert sim.fsm.state == State.ACQUIRE_PERSON
    center, half = sim.fsm.tick(sim.t).gate
    assert center == pytest.approx(0.2) and half == pytest.approx(0.44)
    assert any(line.startswith('DOA->turn: requested +46 deg, turned +34 deg')
               for line in sim.fsm.tick(sim.t).log) is False  # logged once, at the transition


def test_a_rotate_result_for_another_turn_is_ignored():
    sim = turning_sim(target=0.8)
    sim._record(sim.fsm.on_rotate_result(0.3, 0.3, 'reached', sim.t))
    sim.run(1.0)
    assert sim.fsm.state == State.TURN_TO_SOUND


def test_missing_rotate_result_aborts_without_walking():
    sim = turning_sim()
    sim.run(8.0, person=obs(bearing=0.0))
    assert sim.fsm.state == State.IDLE
    assert sim.summaries[-1]['stop_reason'] == 'turn_no_result'
    assert sim.motion_commands() == []


def test_required_direction_missing_aborts_without_walking():
    sim = Sim(skip_turn_to_sound=False, require_direction=True)
    sim.fsm.on_direction(0.9, 0.2, sim.t)                   # heard, but not confident
    sim.wake()
    sim.run(3.0, person=obs(bearing=0.0))
    assert sim.fsm.state == State.IDLE
    assert sim.summaries[-1]['stop_reason'] == 'no_direction'
    assert sim.motion_commands() == [] and sim.rotates == []


def test_a_bearing_from_the_utterance_before_recognition_latency_is_used():
    # Built-in DOA reports only while the caller speaks; Whisper publishes the
    # wake about 2.5 s after speech ends. That bearing must still count.
    sim = Sim(skip_turn_to_sound=False, direction_max_age_s=5.0)
    sim.fsm.on_direction(1.0, 0.9, sim.t)
    sim.run(2.6)
    sim.wake()
    sim.run(0.3)
    assert sim.rotates == [1.0]


def test_a_bearing_older_than_max_age_is_not_used():
    sim = Sim(skip_turn_to_sound=False, direction_max_age_s=5.0)
    sim.fsm.on_direction(1.0, 0.9, sim.t)
    sim.run(5.5)
    sim.wake()
    sim.run(2.0)
    assert sim.rotates == []


def test_speech_after_direction_and_once_at_acquisition():
    sim = turning_sim(target=0.8, wake_speak_text='', direction_speak_text='A|B',
                      acquired_speak_text='C')
    assert sim.says == ['A|B']
    sim._record(sim.fsm.on_rotate_result(0.8, 0.8, 'reached', sim.t))
    sim.run(1.0)
    sim.run(2.0, person=obs(bearing=0.05), every_ticks=1)
    assert sim.fsm.state == State.WALK
    assert sim.says == ['A|B', 'C']
    center, half = sim.fsm.tick(sim.t).gate                 # gate follows the tracked caller
    assert center == pytest.approx(0.05) and half == pytest.approx(0.44)


def test_seated_finish_holds_until_operator_reset():
    sim = walking_sim(arrival_mode=ARRIVAL_SIT_AND_IDENTIFY, sit_hold_until_reset=True,
                      speak_text='Made it.|Here I am.')
    t_arrive = sim.t
    sim.run(0.3, person=obs(bbox=0.8))
    assert sim.fsm.state == State.SIT_AND_IDENTIFY and sim.sits == 0
    assert sim.cmd == (0.0, 0.0)
    sim.run(1.0)
    assert sim.sits == 1 and sim.fsm.display_state == 'SIT'
    sim.run(3.0)
    assert sim.face_requests == 1 and sim.fsm.display_state == 'LOOK_AT_FACE'
    sim._record(sim.fsm.on_face_result(True, sim.t, 0.55))
    sim.run(0.2)
    assert sim.says[-1] == 'Made it.|Here I am.'
    assert sim.fsm.display_state == 'DONE' and sim.stands == 0
    assert sim.summaries[-1]['success'] is True and sim.summaries[-1]['face_center_x'] == 0.55
    sim.run(30.0)
    sim.wake()                                              # still seated: ignored
    assert sim.fsm.state == State.SIT_AND_IDENTIFY and sim.stands == 0
    assert sim.motion_commands(since=t_arrive + 0.3) == []
    sim._record(sim.fsm.on_reset(sim.t))
    assert sim.stands == 1
    sim.run(1.0)
    assert sim.fsm.state == State.IDLE


def test_final_align_turns_onto_the_caller_before_sitting():
    sim = walking_sim(arrival_mode=ARRIVAL_SIT_AND_IDENTIFY, final_align_rad=0.1)
    t0 = sim.t
    sim.run(0.2, person=obs(bearing=0.6, bbox=0.8), every_ticks=1)
    assert sim.fsm.display_state == 'ALIGN_TO_CALLER'
    moves = sim.motion_commands(since=t0)
    assert moves and all(vx == 0.0 and w > 0.0 for _, vx, w in moves)
    assert sim.sits == 0
    sim.run(3.0, person=obs(bearing=0.0, bbox=0.8), every_ticks=1)
    assert sim.sits == 1
    assert_single_axis(sim)


def test_estop_during_the_seated_finish_ends_the_trial():
    sim = walking_sim(arrival_mode=ARRIVAL_SIT_AND_IDENTIFY, sit_hold_until_reset=True)
    sim.run(0.3, person=obs(bbox=0.8))
    sim.estop(True)
    assert sim.fsm.state == State.IDLE and sim.cmd == (0.0, 0.0)


def test_walk_budget_with_caller_centered_arrives_and_sits():
    sim = walking_sim(arrival_mode=ARRIVAL_SIT_AND_IDENTIFY, max_walk_distance_m=1.0,
                      walk_budget_arrives=True)
    sim.run(3.0, person=obs(bearing=0.0, bbox=0.6), every_ticks=1)
    assert sim.fsm.state == State.SIT_AND_IDENTIFY
    sim.run(1.2)
    assert sim.sits == 1


def test_walk_budget_without_the_flag_still_aborts():
    sim = walking_sim(arrival_mode=ARRIVAL_SIT_AND_IDENTIFY, max_walk_distance_m=1.0)
    sim.run(3.0, person=obs(bearing=0.0, bbox=0.6), every_ticks=1)
    assert sim.fsm.state == State.IDLE and sim.sits == 0
    assert sim.summaries[-1]['stop_reason'] == 'walk_budget'


def test_align_by_rotate_turns_once_by_the_bearing_then_walks():
    sim = Sim(align_by_rotate=True)
    sim.wake()
    sim.run(0.6, person=obs(bearing=0.6), every_ticks=1)
    assert sim.fsm.state == State.TURN_TO_SOUND
    assert len(sim.rotates) == 1 and sim.rotates[0] == pytest.approx(0.6, abs=0.05)
    assert all(w == 0.0 for _, _, w in sim.velocities)          # no yaw velocity command
    sim.run(2.0, person=obs(bearing=0.6), every_ticks=1)        # still turning: no second turn
    assert len(sim.rotates) == 1 and sim.motion_commands() == []
    sim._record(sim.fsm.on_rotate_result(sim.rotates[0], 0.58, 'reached', sim.t))
    sim.run(2.0, person=obs(bearing=0.02), every_ticks=1)
    assert sim.fsm.state == State.WALK
    assert_single_axis(sim)


def test_align_by_rotate_gives_up_after_max_turns():
    sim = Sim(align_by_rotate=True, max_align_turns=1)
    sim.wake()
    sim.run(0.6, person=obs(bearing=0.6), every_ticks=1)
    sim._record(sim.fsm.on_rotate_result(sim.rotates[0], 0.0, 'timeout', sim.t))
    sim.run(3.0, person=obs(bearing=0.6), every_ticks=1)
    assert sim.fsm.state == State.IDLE
    assert sim.summaries[-1]['stop_reason'] == 'align_failed'
    assert sim.motion_commands() == []


def test_turn_result_is_matched_to_the_commanded_angle_while_doa_keeps_streaming():
    sim = turning_sim(target=0.99)
    for az in (0.7, 1.2, 0.4):                       # built-in DOA updates mid-turn
        sim.fsm.on_direction(az, 0.9, sim.t)
        sim.run(0.2)
    sim._record(sim.fsm.on_rotate_result(0.99, 0.95, 'reached', sim.t))
    sim.run(1.0)
    assert sim.fsm.state == State.ACQUIRE_PERSON
    assert sim.rotates == [0.99]


def test_search_turn_continues_toward_the_voice_side_when_nobody_is_seen():
    sim = turning_sim(target=0.86, search_turn_rad=0.6, search_turn_after_s=2.0, max_search_turns=2)
    sim._record(sim.fsm.on_rotate_result(0.86, 0.75, 'reached', sim.t))
    sim.run(1.0)
    assert sim.fsm.state == State.ACQUIRE_PERSON
    sim.run(2.2, person=MISS, every_ticks=1)                   # caller still out of view
    assert sim.fsm.state == State.TURN_TO_SOUND
    assert sim.rotates == [0.86, 0.6]                           # same side, fixed step
    sim._record(sim.fsm.on_rotate_result(0.6, 0.58, 'reached', sim.t))
    sim.run(1.0)
    sim.run(1.0, person=obs(bearing=0.05), every_ticks=1)
    assert sim.fsm.state in (State.WALK, State.ALIGN)
    assert sim.summaries == [] and len(sim.rotates) == 2


def test_no_search_turn_when_the_caller_is_already_seen():
    sim = turning_sim(target=-0.86, search_turn_rad=0.6)
    sim._record(sim.fsm.on_rotate_result(-0.86, -0.8, 'reached', sim.t))
    sim.run(1.0)
    sim.run(3.0, person=obs(bearing=0.0), every_ticks=1)
    assert sim.rotates == [-0.86]


def test_search_turns_are_bounded_then_the_trial_ends_without_walking():
    sim = turning_sim(target=-0.86, search_turn_rad=0.6, max_search_turns=1)
    sim._record(sim.fsm.on_rotate_result(-0.86, -0.8, 'reached', sim.t))
    sim.run(3.0, person=MISS, every_ticks=1)
    assert sim.rotates == [-0.86, -0.6]
    sim._record(sim.fsm.on_rotate_result(-0.6, -0.6, 'reached', sim.t))
    sim.run(12.0, person=MISS, every_ticks=1)
    assert sim.fsm.state == State.IDLE and sim.motion_commands() == []
    assert len(sim.rotates) == 2


def test_camera_scan_finds_the_caller_after_a_wrong_voice_bearing():
    """Lab 09-15: voice from the right read +141 deg; the robot turned left and
    2 x 34 deg search turns stopped 66 deg short. 45 deg steps keep scanning the
    same way until the camera sees the caller, and nothing walks before that."""
    sim = turning_sim(target=2.46, search_turn_rad=0.785, search_turn_after_s=2.0,
                      max_search_turns=7)
    sim._record(sim.fsm.on_rotate_result(2.46, 2.446, 'reached', sim.t))
    sim.run(1.0)
    for _ in range(3):                                           # caller still out of view
        sim.run(2.2, person=MISS, every_ticks=1)
        assert sim.fsm.state == State.TURN_TO_SOUND
        assert sim.motion_commands() == []
        sim._record(sim.fsm.on_rotate_result(0.785, 0.76, 'reached', sim.t))
        sim.run(1.0)
    assert sim.rotates == [2.46, 0.785, 0.785, 0.785]            # one direction, fixed step
    sim.run(1.0, person=obs(bearing=0.05), every_ticks=1)
    assert sim.fsm.state in (State.WALK, State.ALIGN)
    assert sim.summaries == []
