"""Deterministic come-here state machine (pure Python, no ROS).

``behavior_node`` is a thin ROS adapter around ``ComeHereFsm``: it feeds sensor
messages and a periodic tick, each with a monotonic timestamp in seconds, and
publishes whatever ``Commands`` come back. Keeping the logic here makes every
transition testable with fake time.

Professor-demo path (``skip_turn_to_sound: true``, ``arrival_mode: stop``)::

    IDLE --wake--> ACQUIRE_PERSON --N fresh detections--> ALIGN or WALK
    ALIGN <--> WALK (hysteresis) --close enough--> ARRIVED --hold--> IDLE

Motion rules, from the GO2 ``mcf`` gait envelope (hardware 2026-04-17/24):
  * ALIGN commands yaw only (vx = 0); WALK commands forward only (yaw = 0).
    No command ever combines the two.
  * ALIGN hands over to WALK once |bearing| is inside the align deadband after
    a minimum hold; WALK re-aligns only after a minimum walk and once |bearing|
    exceeds the larger realign threshold.

Safety rules, all failing closed:
  * a person observation counts only if every field is finite and in range and
    the confidence clears the threshold; anything else is a miss;
  * while moving, a miss lasting ``lost_debounce_s`` or no valid detection for
    ``person_stale_timeout_s`` stops immediately and returns to
    ACQUIRE_PERSON, which needs N new detections before motion resumes;
  * ACQUIRE_PERSON times out to IDLE; the approach also has a time limit and a
    commanded walking-distance budget that stop the robot even if every
    perception-based stop fails;
  * the e-stop aborts the trial, stops, and blocks wake phrases until released.
"""

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    TURN_TO_SOUND = auto()
    ACQUIRE_PERSON = auto()
    ALIGN = auto()
    WALK = auto()
    ARRIVED = auto()
    SIT_AND_IDENTIFY = auto()


MOTION_STATES = (State.ALIGN, State.WALK)

ARRIVAL_STOP = 'stop'
ARRIVAL_SIT_AND_IDENTIFY = 'sit_and_identify'

ARRIVED_REASONS = ('arrived_bbox', 'arrived_distance')

# Tolerance for duration comparisons, so a 0.3 s debounce on a 10 Hz tick is
# exactly three ticks despite float rounding.
_TIME_EPS = 1e-6

# Distance source codes in field 6 of /come_here/person_detection.
DISTANCE_SOURCES = {0: 'none', 1: 'bbox_pinhole', 2: 'lidar'}


@dataclass(frozen=True)
class FsmConfig:
    """Behavior parameters. Field names are the behavior_node ROS parameter names."""

    # Audio direction (experimental, off in the professor demo).
    skip_turn_to_sound: bool = True
    direction_confidence_threshold: float = 0.5
    listening_timeout_s: float = 1.5
    direction_max_age_s: float = 3.0    # a bearing older than this is another utterance
    turn_min_rad: float = 0.2           # caller already ahead: skip the turn
    # Person acquisition.
    person_confidence_threshold: float = 0.5
    search_min_consecutive_detections: int = 2
    search_timeout_s: float = 10.0
    max_person_bearing_rad: float = 1.0
    max_person_distance_m: float = 10.0
    # Lost person.
    lost_debounce_s: float = 0.3
    person_stale_timeout_s: float = 1.5
    # Bearing smoothing: the single EMA layer in the pipeline.
    bearing_ema_alpha: float = 0.3
    # ALIGN / WALK controller (hardware-tuned 2026-04-24).
    approach_align_threshold_rad: float = 0.15
    approach_realign_threshold_rad: float = 0.30
    approach_min_align_s: float = 0.4
    approach_min_walk_s: float = 1.5
    approach_speed: float = 0.6
    approach_ccw_yaw: float = 0.6
    approach_cw_yaw: float = 0.6
    # Stopping.
    approach_stop_distance_m: float = 0.8
    bbox_stop_fraction: float = 0.75
    max_walk_distance_m: float = 4.0
    approach_timeout_s: float = 30.0
    # Arrival.
    arrival_mode: str = ARRIVAL_STOP
    arrival_hold_s: float = 2.0
    wake_speak_text: str = 'I am coming'
    speak_text: str = 'I am here'
    # SIT_AND_IDENTIFY timing (arrival_mode: sit_and_identify).
    sit_settle_s: float = 3.0
    face_timeout_s: float = 1.5
    speak_hold_s: float = 5.0
    stand_settle_s: float = 0.5

    def validate(self) -> None:
        def positive(name):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f'{name} must be finite and > 0, got {value}')

        for name in (
            'listening_timeout_s', 'direction_max_age_s', 'search_timeout_s', 'max_person_bearing_rad',
            'max_person_distance_m', 'person_stale_timeout_s',
            'approach_align_threshold_rad', 'approach_realign_threshold_rad',
            'approach_speed', 'approach_ccw_yaw', 'approach_cw_yaw',
            'approach_stop_distance_m', 'bbox_stop_fraction',
            'max_walk_distance_m', 'approach_timeout_s',
        ):
            positive(name)
        for name in (
            'turn_min_rad', 'lost_debounce_s', 'approach_min_align_s', 'approach_min_walk_s',
            'arrival_hold_s', 'sit_settle_s', 'face_timeout_s', 'speak_hold_s',
            'stand_settle_s',
        ):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f'{name} must be finite and >= 0, got {value}')
        if not 0.0 < self.bearing_ema_alpha <= 1.0:
            raise ValueError('bearing_ema_alpha must be in (0, 1]')
        if not 0.0 <= self.person_confidence_threshold <= 1.0:
            raise ValueError('person_confidence_threshold must be in [0, 1]')
        if self.search_min_consecutive_detections < 1:
            raise ValueError('search_min_consecutive_detections must be >= 1')
        if self.approach_align_threshold_rad >= self.approach_realign_threshold_rad:
            raise ValueError(
                'approach_align_threshold_rad must be < approach_realign_threshold_rad '
                '(hysteresis)'
            )
        if self.approach_realign_threshold_rad >= self.max_person_bearing_rad:
            raise ValueError('approach_realign_threshold_rad must be < max_person_bearing_rad')
        if self.bbox_stop_fraction > 1.0:
            raise ValueError('bbox_stop_fraction > 1 would disable the close-range stop')
        if self.arrival_mode not in (ARRIVAL_STOP, ARRIVAL_SIT_AND_IDENTIFY):
            raise ValueError(f'unknown arrival_mode {self.arrival_mode!r}')


@dataclass(frozen=True)
class PersonObservation:
    """One /come_here/person_detection message (all fields as published floats)."""

    bearing_rad: float
    distance_m: float
    confidence: float
    detected: float
    bbox_h_frac: float = 0.0
    distance_source: float = 0.0
    frame_age_s: float = 0.0

    @classmethod
    def from_array(cls, data: Sequence[float]) -> Optional['PersonObservation']:
        """Parse ``[bearing, distance, confidence, detected, bbox_h_frac,
        distance_source, frame_age_s]``; the last three are optional.
        Returns None for a malformed array."""
        if not 4 <= len(data) <= 7:
            return None
        try:
            values = [float(v) for v in data]
        except (TypeError, ValueError):
            return None
        return cls(*values)


@dataclass
class Commands:
    """Everything the ROS adapter should publish after one FSM call."""

    velocity: Optional[Tuple[float, float]] = None
    rotate_rad: Optional[float] = None
    say: Optional[str] = None
    sit: bool = False
    stand: bool = False
    face_request: bool = False
    trial_summary: Optional[dict] = None
    log: List[str] = field(default_factory=list)


@dataclass
class TrialStats:
    wake_time_s: float
    wake_phrase: str
    first_detection_s: Optional[float] = None
    acquired_s: Optional[float] = None
    acquire_confidence: Optional[float] = None
    approach_start_s: Optional[float] = None
    approach_end_s: Optional[float] = None
    align_phases: int = 0
    walk_phases: int = 0
    lost_events: int = 0
    stale_events: int = 0
    reacquisitions: int = 0
    invalid_observations: int = 0
    min_person_confidence: Optional[float] = None
    max_person_confidence: Optional[float] = None
    max_perception_gap_s: float = 0.0
    commanded_walk_distance_m: float = 0.0
    motion_commands: int = 0
    combined_commands: int = 0
    estop: bool = False
    estop_after_stop: bool = False
    stop_reason: Optional[str] = None
    final_distance_m: Optional[float] = None
    final_distance_source: Optional[str] = None
    final_bbox_h_frac: Optional[float] = None
    final_bearing_rad: Optional[float] = None
    final_person_confidence: Optional[float] = None
    face_present: Optional[bool] = None
    turn_rad: Optional[float] = None       # TURN_TO_SOUND command, if any
    turn_confidence: Optional[float] = None

    def summary(self, end_s: float) -> dict:
        def rel(t):
            return None if t is None else round(t - self.wake_time_s, 3)

        def rnd(v):
            return None if v is None else round(v, 3)

        approach_duration = None
        if self.approach_start_s is not None:
            end = self.approach_end_s if self.approach_end_s is not None else end_s
            approach_duration = round(end - self.approach_start_s, 3)
        abort_stop = self.stop_reason not in ARRIVED_REASONS + (None,)
        return {
            'wake_phrase': self.wake_phrase,
            'first_detection_latency_s': rel(self.first_detection_s),
            'acquire_latency_s': rel(self.acquired_s),
            'acquire_confidence': rnd(self.acquire_confidence),
            'approach_start_s': rel(self.approach_start_s),
            'approach_duration_s': approach_duration,
            'trial_duration_s': round(end_s - self.wake_time_s, 3),
            'align_phases': self.align_phases,
            'walk_phases': self.walk_phases,
            'lost_events': self.lost_events,
            'stale_events': self.stale_events,
            'reacquisitions': self.reacquisitions,
            'safety_stop_events': self.lost_events + self.stale_events + int(abort_stop),
            'invalid_observations': self.invalid_observations,
            'min_person_confidence': rnd(self.min_person_confidence),
            'max_person_confidence': rnd(self.max_person_confidence),
            'max_perception_gap_s': rnd(self.max_perception_gap_s),
            'commanded_walk_distance_m': rnd(self.commanded_walk_distance_m),
            'motion_commands': self.motion_commands,
            'combined_commands': self.combined_commands,
            'estop': self.estop,
            'estop_after_stop': self.estop_after_stop,
            'stop_reason': self.stop_reason,
            'final_distance_m': rnd(self.final_distance_m),
            'final_distance_source': self.final_distance_source,
            'final_bbox_h_frac': rnd(self.final_bbox_h_frac),
            'final_bearing_rad': rnd(self.final_bearing_rad),
            'final_person_confidence': rnd(self.final_person_confidence),
            'turn_rad': rnd(self.turn_rad),
            'turn_confidence': rnd(self.turn_confidence),
            'face_present': self.face_present,
            'success': self.stop_reason in ARRIVED_REASONS and not self.estop,
        }


class ComeHereFsm:
    def __init__(self, config: FsmConfig):
        config.validate()
        self.config = config
        self._state = State.IDLE
        self._state_since = 0.0
        self._estopped = False
        self._trial: Optional[TrialStats] = None

        self._last_azimuth = 0.0
        self._last_dir_confidence = 0.0
        self._last_dir_s: Optional[float] = None

        self._last_person_msg_s: Optional[float] = None
        self._reset_person_tracking()

        self._acquire_since = 0.0
        self._phase_since = 0.0
        self._approach_start: Optional[float] = None
        self._last_cmd_vx = 0.0
        self._vx_since: Optional[float] = None
        self._walk_distance_m = 0.0

        self._sit_substep = 0
        self._sit_step_since = 0.0
        self._face_received = False

    # -- read-only state --

    @property
    def state(self) -> State:
        return self._state

    @property
    def estopped(self) -> bool:
        return self._estopped

    @property
    def trial_active(self) -> bool:
        return self._trial is not None

    def status(self) -> dict:
        """Compact snapshot for operator logs; never used for decisions."""
        obs = self._last_valid_obs
        return {
            'state': self._state.name,
            'bearing': None if self._ema_bearing is None else round(self._ema_bearing, 2),
            'bbox_h_frac': None if obs is None else round(obs.bbox_h_frac, 2),
            'distance_m': None if obs is None else round(obs.distance_m, 2),
            'walked_m': round(self._walk_distance_m, 2),
            'cmd_vx': self._last_cmd_vx,
        }

    # -- inputs --

    def on_wake(self, phrase: str, now: float) -> Commands:
        cmds = Commands()
        if self._estopped:
            cmds.log.append(f'Wake "{phrase}" ignored: e-stop engaged')
            return cmds
        if self._state != State.IDLE:
            return cmds
        cfg = self.config
        self._trial = TrialStats(wake_time_s=now, wake_phrase=phrase)
        self._reset_person_tracking()
        self._approach_start = None
        self._walk_distance_m = 0.0
        # An explicit stop first: keeps the robot still and re-arms the bridge
        # after an e-stop release, so motion only ever resumes on a new trial.
        self._command(cmds, now, 0.0, 0.0)
        if cfg.wake_speak_text:
            cmds.say = cfg.wake_speak_text
        cmds.log.append(f'Wake phrase "{phrase}" accepted')
        if cfg.skip_turn_to_sound:
            self._enter(State.ACQUIRE_PERSON, now, cmds, 'wake')
        else:
            self._enter(State.LISTENING, now, cmds, 'wake')
        return cmds

    def on_direction(self, azimuth_rad: float, confidence: float, now: float) -> Commands:
        cmds = Commands()
        if not (math.isfinite(azimuth_rad) and math.isfinite(confidence)):
            cmds.log.append('Ignoring non-finite audio_direction')
            return cmds
        if abs(azimuth_rad) > math.pi:
            cmds.log.append(f'Ignoring out-of-range azimuth {azimuth_rad:.2f} rad')
            return cmds
        self._last_azimuth = azimuth_rad
        self._last_dir_confidence = confidence
        self._last_dir_s = now
        return cmds

    def on_person(self, obs: Optional[PersonObservation], now: float) -> Commands:
        cmds = Commands()
        trial = self._trial
        if trial is not None and self._last_person_msg_s is not None:
            trial.max_perception_gap_s = max(
                trial.max_perception_gap_s, now - self._last_person_msg_s
            )
        self._last_person_msg_s = now

        kind = self.classify(obs)
        if kind == 'positive':
            self._consec_hits += 1
            self._last_valid_s = now
            self._last_valid_obs = obs
            self._miss_since = None
            alpha = self.config.bearing_ema_alpha
            if self._ema_bearing is None or self._ema_reseed:
                self._ema_bearing = obs.bearing_rad
            else:
                self._ema_bearing = alpha * obs.bearing_rad + (1.0 - alpha) * self._ema_bearing
            self._ema_reseed = False
            if trial is not None:
                if trial.first_detection_s is None:
                    trial.first_detection_s = now
                c = obs.confidence
                trial.min_person_confidence = (
                    c if trial.min_person_confidence is None
                    else min(trial.min_person_confidence, c)
                )
                trial.max_person_confidence = (
                    c if trial.max_person_confidence is None
                    else max(trial.max_person_confidence, c)
                )
        else:
            if kind == 'invalid' and trial is not None:
                trial.invalid_observations += 1
            self._consec_hits = 0
            # The next valid detection seeds the EMA fresh instead of blending
            # with a pre-gap bearing.
            self._ema_reseed = True
            if self._miss_since is None:
                self._miss_since = now
        return cmds

    def on_face_result(self, face_present: bool, now: float) -> Commands:
        if self._state == State.SIT_AND_IDENTIFY and self._sit_substep == 1:
            self._face_received = True
            if self._trial is not None:
                self._trial.face_present = bool(face_present)
        return Commands()

    def on_estop(self, engaged: bool, now: float) -> Commands:
        cmds = Commands()
        if engaged:
            newly_engaged = not self._estopped
            self._estopped = True
            self._command(cmds, now, 0.0, 0.0)
            if self._trial is not None:
                if self._trial.stop_reason is None:
                    self._trial.stop_reason = 'estop'
                    self._trial.estop = True
                    self._record_final()
                else:
                    self._trial.estop_after_stop = True
            if self._state != State.IDLE:
                self._finish(now, cmds, 'estop')
            if newly_engaged:
                cmds.log.append('E-stop engaged: stopped, wake phrases blocked')
        elif self._estopped:
            self._estopped = False
            cmds.log.append('E-stop released')
        return cmds

    def shutdown(self, now: float) -> Commands:
        cmds = Commands()
        self._command(cmds, now, 0.0, 0.0)
        if self._trial is not None:
            if self._trial.stop_reason is None:
                self._trial.stop_reason = 'shutdown'
                self._record_final()
            cmds.trial_summary = self._trial.summary(now)
            self._trial = None
        self._state = State.IDLE
        return cmds

    def tick(self, now: float) -> Commands:
        cmds = Commands()
        self._integrate_walk(now)
        state = self._state
        if state == State.IDLE:
            return cmds
        if state == State.LISTENING:
            self._tick_listening(now, cmds)
        elif state == State.TURN_TO_SOUND:
            cmds.rotate_rad = self._last_azimuth
            if self._trial is not None:
                self._trial.turn_rad = self._last_azimuth
                self._trial.turn_confidence = self._last_dir_confidence
            cmds.log.append(f'Rotating toward sound: {self._last_azimuth:+.2f} rad '
                            f'(confidence {self._last_dir_confidence:.2f})')
            self._enter(State.ACQUIRE_PERSON, now, cmds, 'rotate sent')
        elif state == State.ACQUIRE_PERSON:
            self._tick_acquire(now, cmds)
        elif state in MOTION_STATES:
            self._tick_motion(now, cmds)
        elif state == State.ARRIVED:
            if now - self._state_since + _TIME_EPS >= self.config.arrival_hold_s:
                self._finish(now, cmds, 'arrival hold complete')
        elif state == State.SIT_AND_IDENTIFY:
            self._tick_sit(now, cmds)
        return cmds

    # -- perception validity --

    def classify(self, obs: Optional[PersonObservation]) -> str:
        """Return 'positive', 'negative' (a normal miss) or 'invalid'."""
        cfg = self.config
        if obs is None:
            return 'invalid'
        values = (
            obs.bearing_rad, obs.distance_m, obs.confidence, obs.detected,
            obs.bbox_h_frac, obs.distance_source, obs.frame_age_s,
        )
        if not all(math.isfinite(v) for v in values):
            return 'invalid'
        if obs.detected not in (0.0, 1.0):
            return 'invalid'
        if obs.detected == 0.0:
            return 'negative'
        if not 0.0 <= obs.confidence <= 1.0:
            return 'invalid'
        if obs.confidence < cfg.person_confidence_threshold:
            return 'negative'
        if abs(obs.bearing_rad) > cfg.max_person_bearing_rad:
            return 'invalid'
        if not 0.0 <= obs.distance_m <= cfg.max_person_distance_m:
            return 'invalid'
        if not 0.0 <= obs.bbox_h_frac <= 1.0 + 1e-6:
            return 'invalid'
        return 'positive'

    # -- state handlers --

    def _tick_listening(self, now: float, cmds: Commands) -> None:
        cfg = self.config
        fresh = (self._last_dir_s is not None
                 and now - self._last_dir_s <= cfg.direction_max_age_s)
        if fresh and self._last_dir_confidence >= cfg.direction_confidence_threshold:
            if abs(self._last_azimuth) < cfg.turn_min_rad:
                self._enter(State.ACQUIRE_PERSON, now, cmds,
                            f'sound ahead ({self._last_azimuth:+.2f} rad), no turn')
            else:
                self._enter(State.TURN_TO_SOUND, now, cmds, 'direction confident')
        elif now - self._state_since > cfg.listening_timeout_s:
            self._enter(State.ACQUIRE_PERSON, now, cmds, 'no confident direction')

    def _tick_acquire(self, now: float, cmds: Commands) -> None:
        cfg = self.config
        trial = self._trial
        if (self._consec_hits >= cfg.search_min_consecutive_detections
                and self._person_fresh(now)):
            if trial is not None:
                if trial.acquired_s is None:
                    trial.acquired_s = now
                    trial.acquire_confidence = self._last_valid_obs.confidence
                else:
                    trial.reacquisitions += 1
            reason = self._close_enough_reason()
            if reason is not None:
                self._arrive(now, cmds, reason)
            elif abs(self._ema_bearing) < cfg.approach_align_threshold_rad:
                self._enter(State.WALK, now, cmds, 'person acquired, aligned')
                self._command(cmds, now, cfg.approach_speed, 0.0)
            else:
                self._enter(State.ALIGN, now, cmds, 'person acquired, misaligned')
                self._command(cmds, now, 0.0, self._yaw_toward(self._ema_bearing))
            return
        if now - self._acquire_since > cfg.search_timeout_s:
            reason = 'acquire_timeout' if self._approach_start is None else 'reacquire_timeout'
            self._abort(now, cmds, reason)

    def _tick_motion(self, now: float, cmds: Commands) -> None:
        cfg = self.config
        trial = self._trial

        if not self._person_fresh(now):
            if trial is not None:
                trial.stale_events += 1
            self._lose(now, cmds, 'no valid detection within person_stale_timeout_s')
            return
        if (self._miss_since is not None
                and now - self._miss_since + _TIME_EPS >= cfg.lost_debounce_s):
            if trial is not None:
                trial.lost_events += 1
            self._lose(now, cmds, 'person lost for lost_debounce_s')
            return

        reason = self._close_enough_reason()
        if reason is not None:
            self._arrive(now, cmds, reason)
            return
        if now - self._approach_start > cfg.approach_timeout_s:
            self._abort(now, cmds, 'approach_timeout')
            return
        if self._walk_distance_m >= cfg.max_walk_distance_m:
            self._abort(now, cmds, 'walk_budget')
            return

        bearing = self._ema_bearing
        elapsed = now - self._phase_since
        if self._state == State.ALIGN:
            if (abs(bearing) < cfg.approach_align_threshold_rad
                    and elapsed + _TIME_EPS >= cfg.approach_min_align_s):
                self._enter(State.WALK, now, cmds, f'aligned bearing={bearing:+.2f}')
                self._command(cmds, now, cfg.approach_speed, 0.0)
            else:
                self._command(cmds, now, 0.0, self._yaw_toward(bearing))
        else:
            if (elapsed + _TIME_EPS >= cfg.approach_min_walk_s
                    and abs(bearing) > cfg.approach_realign_threshold_rad):
                self._enter(State.ALIGN, now, cmds, f'realign bearing={bearing:+.2f}')
                self._command(cmds, now, 0.0, self._yaw_toward(bearing))
            else:
                self._command(cmds, now, cfg.approach_speed, 0.0)

    def _tick_sit(self, now: float, cmds: Commands) -> None:
        cfg = self.config
        elapsed = now - self._sit_step_since + _TIME_EPS
        if self._sit_substep == 0 and elapsed >= cfg.sit_settle_s:
            cmds.face_request = True
            self._next_sit_substep(now)
        elif self._sit_substep == 1 and (self._face_received or elapsed >= cfg.face_timeout_s):
            if cfg.speak_text:
                cmds.say = cfg.speak_text
            self._next_sit_substep(now)
        elif self._sit_substep == 2 and elapsed >= cfg.speak_hold_s:
            cmds.stand = True
            self._next_sit_substep(now)
        elif self._sit_substep == 3 and elapsed >= cfg.stand_settle_s:
            self._finish(now, cmds, 'sit sequence complete')

    # -- transitions --

    def _enter(self, new_state: State, now: float, cmds: Commands, reason: str) -> None:
        old = self._state
        self._state = new_state
        self._state_since = now
        cmds.log.append(f'State: {old.name} -> {new_state.name} ({reason})')
        trial = self._trial
        if new_state == State.ACQUIRE_PERSON:
            self._acquire_since = now
            self._consec_hits = 0
        elif new_state in MOTION_STATES:
            self._phase_since = now
            if self._approach_start is None:
                self._approach_start = now
                if trial is not None:
                    trial.approach_start_s = now
            if trial is not None:
                if new_state == State.ALIGN:
                    trial.align_phases += 1
                else:
                    trial.walk_phases += 1

    def _lose(self, now: float, cmds: Commands, why: str) -> None:
        self._command(cmds, now, 0.0, 0.0)
        self._enter(State.ACQUIRE_PERSON, now, cmds, why)

    def _arrive(self, now: float, cmds: Commands, reason: str) -> None:
        self._command(cmds, now, 0.0, 0.0)
        if self._trial is not None:
            self._trial.stop_reason = reason
            self._trial.approach_end_s = now
            self._record_final()
        obs = self._last_valid_obs
        cmds.log.append(
            f'Close enough ({reason}): bbox_h_frac={obs.bbox_h_frac:.2f} '
            f'distance={obs.distance_m:.2f} m'
        )
        if self.config.arrival_mode == ARRIVAL_SIT_AND_IDENTIFY:
            self._enter(State.SIT_AND_IDENTIFY, now, cmds, reason)
            cmds.sit = True
            self._sit_substep = 0
            self._sit_step_since = now
            self._face_received = False
        else:
            self._enter(State.ARRIVED, now, cmds, reason)
            if self.config.speak_text:
                cmds.say = self.config.speak_text

    def _abort(self, now: float, cmds: Commands, reason: str) -> None:
        self._command(cmds, now, 0.0, 0.0)
        if self._trial is not None and self._trial.stop_reason is None:
            self._trial.stop_reason = reason
            if self._approach_start is not None:
                self._trial.approach_end_s = now
            self._record_final()
        self._finish(now, cmds, reason)

    def _finish(self, now: float, cmds: Commands, reason: str) -> None:
        if self._trial is not None:
            cmds.trial_summary = self._trial.summary(now)
            self._trial = None
        self._enter(State.IDLE, now, cmds, reason)

    def _next_sit_substep(self, now: float) -> None:
        self._sit_substep += 1
        self._sit_step_since = now

    # -- helpers --

    def _reset_person_tracking(self) -> None:
        self._consec_hits = 0
        self._last_valid_s: Optional[float] = None
        self._last_valid_obs: Optional[PersonObservation] = None
        self._miss_since: Optional[float] = None
        self._ema_bearing: Optional[float] = None
        self._ema_reseed = False

    def _person_fresh(self, now: float) -> bool:
        return (
            self._last_valid_s is not None
            and now - self._last_valid_s <= self.config.person_stale_timeout_s + _TIME_EPS
        )

    def _close_enough_reason(self) -> Optional[str]:
        obs = self._last_valid_obs
        if obs is None:
            return None
        if obs.bbox_h_frac >= self.config.bbox_stop_fraction:
            return 'arrived_bbox'
        if 0.0 < obs.distance_m <= self.config.approach_stop_distance_m:
            return 'arrived_distance'
        return None

    def _yaw_toward(self, bearing: float) -> float:
        # Positive bearing = person to the left = counter-clockwise yaw.
        if bearing > 0.0:
            return self.config.approach_ccw_yaw
        return -self.config.approach_cw_yaw

    def _integrate_walk(self, now: float) -> None:
        """Dead-reckon commanded forward distance (the walk-budget backstop)."""
        if self._vx_since is not None and self._last_cmd_vx > 0.0:
            self._walk_distance_m += self._last_cmd_vx * max(0.0, now - self._vx_since)
            if self._trial is not None:
                self._trial.commanded_walk_distance_m = self._walk_distance_m
        self._vx_since = now

    def _command(self, cmds: Commands, now: float, vx: float, yaw_rate: float) -> None:
        self._integrate_walk(now)
        cmds.velocity = (float(vx), float(yaw_rate))
        self._last_cmd_vx = float(vx)
        trial = self._trial
        if trial is not None and (vx != 0.0 or yaw_rate != 0.0):
            trial.motion_commands += 1
            if vx != 0.0 and yaw_rate != 0.0:
                trial.combined_commands += 1

    def _record_final(self) -> None:
        trial = self._trial
        obs = self._last_valid_obs
        if trial is None or obs is None:
            return
        trial.final_distance_m = obs.distance_m
        trial.final_distance_source = DISTANCE_SOURCES.get(int(obs.distance_source), 'unknown')
        trial.final_bbox_h_frac = obs.bbox_h_frac
        trial.final_bearing_rad = self._ema_bearing
        trial.final_person_confidence = obs.confidence
