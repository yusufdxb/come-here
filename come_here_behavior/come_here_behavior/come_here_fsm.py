"""Deterministic come-here state machine (pure Python, no ROS).

``behavior_node`` is a thin ROS adapter around ``ComeHereFsm``: it feeds sensor
messages and a periodic tick, each with a monotonic timestamp in seconds, and
publishes whatever ``Commands`` come back. Keeping the logic here makes every
transition testable with fake time.

Class-demo path (``skip_turn_to_sound: false``, ``arrival_mode: sit_and_identify``)::

    IDLE --wake--> LISTENING (LOCALIZE_SOUND) --confident DOA--> TURN_TO_SOUND
    TURN_TO_SOUND: one rotate command, wait for the bridge's rotate_result
    --> ACQUIRE_PERSON with a DOA gate: perception only reports a person whose
        bearing is within acquire_gate_half_rad of where the voice now is
    --> ALIGN <--> WALK (hysteresis, gate follows the tracked person)
    --close enough--> SIT_AND_IDENTIFY: stop, settle, sit, face check, speak,
        stay seated (DONE) until the operator reset

Camera-only path (``skip_turn_to_sound: true``, ``arrival_mode: stop``)::

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

ARRIVED_REASONS = ('arrived_bbox', 'arrived_distance', 'arrived_walk_budget')

# Stepwise skill interface (behavior_node parameter enable_skill_api; off by default).
# Each skill is one action whose result a supervisor reads before choosing the next;
# a step that needs an earlier one names that step's request_id (a handoff token).
SKILL_API_VERSION = 2
SKILL_LOCALIZE = 'localize_caller'       # no motion: a fresh, confident voice bearing
SKILL_ORIENT = 'orient_to_caller'        # one turn by a localization's bearing
SKILL_ACQUIRE = 'acquire_caller'         # no motion: the caller in the camera, gated
SKILL_APPROACH = 'approach_caller'       # walk to an acquired caller; ends STANDING
SKILL_SIT = 'sit'                        # sit after an arrival, nothing else
SKILL_ASK_AGAIN = 'ask_caller_again'     # no motion: ask the caller to speak again
SKILL_CANCEL = 'cancel'
SKILLS = (SKILL_LOCALIZE, SKILL_ORIENT, SKILL_ACQUIRE, SKILL_APPROACH, SKILL_SIT,
          SKILL_ASK_AGAIN, SKILL_CANCEL)
SKILL_ARGS = {SKILL_LOCALIZE: set(), SKILL_ORIENT: {'localization'},
              SKILL_ACQUIRE: set(), SKILL_APPROACH: {'acquisition'},
              SKILL_SIT: {'arrival'}, SKILL_ASK_AGAIN: set(), SKILL_CANCEL: {'request_id'}}
SKILL_MOTION = (SKILL_ORIENT, SKILL_APPROACH, SKILL_SIT)
SKILL_TURN_OK = ('turn_not_needed', 'reached', 'overshoot', 'timed')
# Sit sequence phases where a cancel cannot stop anything: the robot is sitting or
# changing posture, and the operator reset owns the stand-up.
SKILL_POSTURE_PHASES = ('sit', 'look', 'speak_hold', 'done', 'stand')
MAX_GOAL_ID_LEN = 64
GOAL_ID_MEMORY = 1024
# Handoff lifetimes: an acquisition holds while the caller stays continuously in view
# (person_stale_timeout_s) and at most this long; an arrival while the robot has not
# moved since.
SKILL_ACQUISITION_MAX_S = 15.0
SKILL_ARRIVAL_VALID_S = 30.0

# Tolerance for duration comparisons, so a 0.3 s debounce on a 10 Hz tick is
# exactly three ticks despite float rounding.
_TIME_EPS = 1e-6

# Distance source codes in field 6 of /come_here/person_detection.
DISTANCE_SOURCES = {0: 'none', 1: 'bbox_pinhole', 2: 'lidar'}

# A bearing must be measured no earlier than this before the wake it belongs to
# (audio_node publishes the direction a few ms before the wake phrase).
WAKE_DIRECTION_SLACK_S = 1.0

# Operator-facing names for the demo sequence (status topic and viewer only).
DISPLAY_NAMES = {
    'IDLE': 'IDLE', 'LISTENING': 'LOCALIZE_SOUND', 'TURN_TO_SOUND': 'ROTATE_TO_DOA',
    'ACQUIRE_PERSON': 'ACQUIRE_PERSON', 'ALIGN': 'ALIGN', 'WALK': 'APPROACH',
    'ARRIVED': 'ARRIVED',
}
SIT_PHASE_NAMES = {
    'final_align': 'ALIGN_TO_CALLER', 'settle': 'ARRIVED', 'sit': 'SIT',
    'look': 'LOOK_AT_FACE', 'speak_hold': 'DONE', 'done': 'DONE', 'stand': 'STAND',
}


def _valid_id(value) -> bool:
    return (isinstance(value, str) and 1 <= len(value) <= MAX_GOAL_ID_LEN
            and value.isprintable())


def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class FsmConfig:
    """Behavior parameters. Field names are the behavior_node ROS parameter names."""

    # Audio direction (experimental, off in the professor demo).
    skip_turn_to_sound: bool = True
    direction_confidence_threshold: float = 0.5
    listening_timeout_s: float = 1.5
    direction_max_age_s: float = 3.0    # a bearing older than this is another utterance
    turn_min_rad: float = 0.2           # caller already ahead: skip the turn
    require_direction: bool = False     # true: no confident DOA -> abort, never guess a caller
    direction_speak_text: str = ''      # said once the voice direction is known ('|' = choices)
    turn_result_timeout_s: float = 7.0  # no rotate_result by then -> abort (fail closed)
    turn_settle_s: float = 0.7          # gate stays closed this long after the turn ends
    # DOA-gated person selection (applied by perception_node via /come_here/target_gate).
    acquire_gate_half_rad: float = 0.44  # +/- 25 deg around the expected caller bearing
    track_gate_half_rad: float = 0.44    # +/- 25 deg around the tracked bearing while moving
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
    walk_budget_arrives: bool = False   # budget reached with the caller tracked and centered = arrival
    align_by_rotate: bool = False       # misaligned caller: one odometry turn by the bearing, not yaw velocity
    max_align_turns: int = 2            # per trial; beyond it a misaligned caller aborts
    # Voice DOA gives the side; the camera finishes the turn. After a DOA turn,
    # nobody seen for search_turn_after_s -> one more turn of search_turn_rad the
    # same way (lab 09-14: caller at 90 deg read +49 deg). 0 disables.
    search_turn_rad: float = 0.0
    search_turn_after_s: float = 2.0
    max_search_turns: int = 2
    # Nobody seen relisten_after_s after the voice turn: say this, go back to
    # LISTENING and turn toward the next wake's bearing, before any search turn
    # ('' disables, '|' = choices). Must not contain the wake phrase, or the
    # robot can wake itself. Once used up, the search turns run as before.
    relisten_speak_text: str = ''
    relisten_after_s: float = 2.0       # upper bound, also covers a dead camera
    # Ask sooner: this many consecutive fresh camera frames with nobody in them
    # (0 = time only). Fast camera -> fast prompt; slow camera -> relisten_after_s.
    relisten_empty_frames: int = 0
    relisten_frame_max_age_s: float = 1.0   # older frames are a stale camera, not "nobody"
    max_relistens: int = 1              # per trial
    relisten_timeout_s: float = 8.0     # no new confident bearing by then -> give up
    approach_timeout_s: float = 30.0
    # Arrival.
    arrival_mode: str = ARRIVAL_STOP
    arrival_hold_s: float = 2.0
    wake_speak_text: str = 'I am coming'
    # Hold still this long after the wake before the first turn, so the wake
    # phrase is heard before the robot moves. 0 = turn immediately.
    wake_speak_hold_s: float = 0.0
    speak_text: str = 'I am here'
    acquired_speak_text: str = ''       # said at the first visual acquisition ('|' = choices)
    # SIT_AND_IDENTIFY timing (arrival_mode: sit_and_identify).
    sit_settle_s: float = 3.0
    face_timeout_s: float = 1.5
    speak_hold_s: float = 5.0
    stand_settle_s: float = 0.5
    final_align_rad: float = 0.0        # > 0: yaw-only turn onto the caller before sitting
    final_align_timeout_s: float = 2.0
    pre_sit_settle_s: float = 1.0       # StopMove, then this long before Sit
    sit_hold_until_reset: bool = False  # true: stay seated after speaking until on_reset

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
            'max_walk_distance_m', 'approach_timeout_s', 'turn_result_timeout_s',
            'acquire_gate_half_rad', 'track_gate_half_rad', 'relisten_timeout_s',
            'relisten_frame_max_age_s',
        ):
            positive(name)
        for name in (
            'turn_min_rad', 'lost_debounce_s', 'approach_min_align_s', 'approach_min_walk_s',
            'arrival_hold_s', 'sit_settle_s', 'face_timeout_s', 'speak_hold_s',
            'stand_settle_s', 'turn_settle_s', 'final_align_rad', 'final_align_timeout_s',
            'pre_sit_settle_s', 'wake_speak_hold_s', 'relisten_after_s',
        ):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f'{name} must be finite and >= 0, got {value}')
        if not 0.0 < self.bearing_ema_alpha <= 1.0:
            raise ValueError('bearing_ema_alpha must be in (0, 1]')
        if not 0.0 <= self.person_confidence_threshold <= 1.0:
            raise ValueError('person_confidence_threshold must be in [0, 1]')
        if self.max_search_turns < 0 or not (math.isfinite(self.search_turn_rad)
                                             and 0.0 <= self.search_turn_rad <= math.pi):
            raise ValueError('search_turn_rad must be in [0, pi] and max_search_turns >= 0')
        if self.max_align_turns < 0:
            raise ValueError('max_align_turns must be >= 0')
        if self.max_relistens < 0 or self.relisten_empty_frames < 0:
            raise ValueError('max_relistens and relisten_empty_frames must be >= 0')
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
    # (center_rad, half_width_rad) for perception's person selection; (x, 0.0) = closed.
    gate: Optional[Tuple[float, float]] = None
    trial_summary: Optional[dict] = None
    # Skill interface results, one per request (always empty on the baseline path).
    skill_results: List[dict] = field(default_factory=list)
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
    turn_turned_rad: Optional[float] = None  # odometry yaw change reported by the bridge
    turn_result_reason: Optional[str] = None
    gate_center_rad: Optional[float] = None  # expected caller bearing in the camera after the turn
    acquire_bearing_rad: Optional[float] = None  # first visual bearing of the selected caller
    face_center_x: Optional[float] = None
    relistens: int = 0
    goal_id: Optional[str] = None           # skill interface only
    skill: Optional[str] = None
    request_id: Optional[str] = None

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
        out = {
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
            'turn_turned_rad': rnd(self.turn_turned_rad),
            'turn_result_reason': self.turn_result_reason,
            'gate_center_rad': rnd(self.gate_center_rad),
            'acquire_bearing_rad': rnd(self.acquire_bearing_rad),
            'face_present': self.face_present,
            'face_center_x': rnd(self.face_center_x),
            'relistens': self.relistens,
            'success': self.stop_reason in ARRIVED_REASONS and not self.estop,
        }
        if self.goal_id is not None:          # baseline trials keep the exact key set
            out['goal_id'] = self.goal_id
            out['skill'] = self.skill
            out['request_id'] = self.request_id
        return out


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

        self._sit_phase = 'settle'
        self._sit_step_since = 0.0
        self._sit_align_sign = 0.0
        self._face_received = False
        self._last_face = None

        self._wake_s: Optional[float] = None
        self._gate_center = 0.0
        self._turn_sent = False
        self._turn_result: Optional[Tuple[float, Optional[float], str]] = None
        self._turn_done_s: Optional[float] = None
        self._turn_target = 0.0
        self._search_sign = 0.0
        self._search_turns = 0
        self._turn_purpose = 'doa'
        self._align_turns = 0
        self._search_sign = 0.0
        self._search_turns = 0
        self._relistens = 0
        self._relisten_s: Optional[float] = None
        self._empty_frames = 0

        # Skill interface state; None / False on every baseline path.
        self._skill: Optional[str] = None
        self._goal_id: Optional[str] = None
        self._request_id: Optional[str] = None
        self._turn_gate_valid = False
        self._seen_request_ids: List[str] = []
        self._retired_goal_ids: List[str] = []
        self._session_goal: Optional[str] = None
        self._session_seq = 0
        # Handoff tokens: {'id', 'goal', 't', ...} from the last successful step.
        self._localization: Optional[dict] = None
        self._acquisition: Optional[dict] = None
        self._arrival: Optional[dict] = None
        self._skill_turn_rad = 0.0
        self._last_motion_s: Optional[float] = None   # last nonzero command or turn end
        self._bearing_floor_s: Optional[float] = None  # a bearing must be newer than this
        self._lost_bearing: Optional[Tuple[float, float]] = None  # (bearing, t) at track loss
        self._skill_seated = False    # a skill sit was sent; cleared by the operator reset
        self._skill_data: dict = {}
        self._pending_localization: dict = {}
        self._skill_turn_conf = 0.0
        self._asked_s: Optional[float] = None

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

    @property
    def display_state(self) -> str:
        if self._state == State.SIT_AND_IDENTIFY:
            return SIT_PHASE_NAMES.get(self._sit_phase, self._sit_phase.upper())
        return DISPLAY_NAMES.get(self._state.name, self._state.name)

    def viewer_status(self, now: float) -> dict:
        """Everything the operator view draws; never used for decisions."""
        obs = self._last_valid_obs
        gate = self._current_gate()
        dir_age = None if self._last_dir_s is None else round(now - self._last_dir_s, 2)
        trial = self._trial
        status = {
            'state': self._state.name,
            'phase': self.display_state,
            'estopped': self._estopped,
            'doa_deg': round(math.degrees(self._last_azimuth), 1),
            'doa_conf': round(self._last_dir_confidence, 2),
            'doa_age_s': dir_age,
            'turn_target_deg': None if trial is None or trial.turn_rad is None
            else round(math.degrees(trial.turn_rad), 1),
            'turn_turned_deg': None if trial is None or trial.turn_turned_rad is None
            else round(math.degrees(trial.turn_turned_rad), 1),
            'gate_center_deg': None if gate is None else round(math.degrees(gate[0]), 1),
            'gate_half_deg': None if gate is None else round(math.degrees(gate[1]), 1),
            'target_bearing_deg': None if self._ema_bearing is None
            else round(math.degrees(self._ema_bearing), 1),
            'target_conf': None if obs is None else round(obs.confidence, 2),
            'target_distance_m': None if obs is None else round(obs.distance_m, 2),
            'target_distance_source': None if obs is None
            else DISTANCE_SOURCES.get(int(obs.distance_source), 'unknown'),
            'target_bbox_h_frac': None if obs is None else round(obs.bbox_h_frac, 2),
            'target_fresh': self._person_fresh(now),
            'walked_m': round(self._walk_distance_m, 2),
            'face': self._last_face,
        }
        if self._skill is not None:
            status['goal_id'] = self._goal_id
            status['skill'] = self._skill
            status['request_id'] = self._request_id
        if self._session_goal is not None:
            status['skill_session'] = {
                'goal_id': self._session_goal, 'seq': self._session_seq,
                'localization': None if self._localization is None
                else self._localization['id'],
                'acquisition': None if self._acquisition is None
                else self._acquisition['id'],
                'arrival': None if self._arrival is None else self._arrival['id'],
                'seated': self._skill_seated,
                'last_motion_age_s': None if self._last_motion_s is None
                else round(now - self._last_motion_s, 2),
            }
        return status

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
        if self._session_goal is not None:
            cmds.log.append(f'Wake "{phrase}" ignored: the skill interface owns this robot')
            return cmds
        cfg = self.config
        self._trial = TrialStats(wake_time_s=now, wake_phrase=phrase)
        self._reset_person_tracking()
        self._approach_start = None
        self._walk_distance_m = 0.0
        self._wake_s = now
        self._gate_center = 0.0
        self._turn_sent = False
        self._turn_result = None
        self._turn_done_s = None
        self._last_face = None
        self._turn_purpose = 'doa'
        self._align_turns = 0
        self._search_sign = 0.0
        self._search_turns = 0
        self._search_sign = 0.0
        self._search_turns = 0
        self._relistens = 0
        self._relisten_s = None
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
        if self._state == State.ACQUIRE_PERSON:
            if kind == 'positive':
                self._empty_frames = 0
            elif (kind == 'negative'
                  and obs.frame_age_s <= self.config.relisten_frame_max_age_s):
                self._empty_frames += 1
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

    def on_face_result(self, face_present: bool, now: float,
                       center_x: Optional[float] = None) -> Commands:
        cmds = Commands()
        if self._state == State.SIT_AND_IDENTIFY and self._sit_phase == 'look':
            self._face_received = True
            cx = center_x if (center_x is not None and math.isfinite(center_x)) else None
            self._last_face = {'present': bool(face_present),
                               'center_x': None if cx is None else round(cx, 3)}
            if self._trial is not None:
                self._trial.face_present = bool(face_present)
                self._trial.face_center_x = cx if face_present else None
            if face_present and cx is not None:
                cmds.log.append(f'LOOK_AT_FACE: face at x={cx:.2f} of the image '
                                f'({"centered" if abs(cx - 0.5) <= 0.2 else "off-center"})')
            else:
                cmds.log.append('LOOK_AT_FACE: no face found in the frame')
        return cmds

    def on_rotate_result(self, target_rad: float, turned_rad: Optional[float], reason: str,
                         now: float) -> Commands:
        """The bridge finished the turn this trial asked for."""
        cmds = Commands()
        if self._state != State.TURN_TO_SOUND or not self._turn_sent or self._turn_result:
            return cmds
        if not math.isfinite(target_rad) or abs(target_rad - self._turn_target) > 0.02:
            cmds.log.append(f'Ignoring rotate_result for another turn (target {target_rad:+.2f})')
            return cmds
        if turned_rad is not None and not math.isfinite(turned_rad):
            turned_rad = None
        self._turn_result = (target_rad, turned_rad, str(reason))
        self._turn_done_s = now
        return cmds

    def on_reset(self, now: float) -> Commands:
        """Operator: stand up from DONE and go back to IDLE."""
        cmds = Commands()
        if self._state == State.SIT_AND_IDENTIFY and self._sit_phase == 'done':
            cmds.stand = True
            self._skill_seated = False
            self._sit_phase = 'stand'
            self._sit_step_since = now
            cmds.log.append('Operator reset: standing up')
        elif self._skill_seated and self._state == State.IDLE:
            # E-stopped while seated: the operator stood the robot up by hand.
            self._skill_seated = False
            cmds.log.append('Operator reset: skill sit cleared (robot stood up by the operator)')
        else:
            cmds.log.append(f'Operator reset ignored in {self.display_state}')
        return cmds

    # -- stepwise skill interface --

    @property
    def active_goal_id(self) -> Optional[str]:
        return self._goal_id if self._skill is not None else None

    @property
    def active_request_id(self) -> Optional[str]:
        return self._request_id if self._skill is not None else None

    def on_skill_request(self, req, now: float) -> Commands:
        """One request from the skill interface (already JSON-decoded).

        Every request gets exactly one result naming its goal_id and request_id. A
        rejection never touches motion: it returns Commands with no velocity, rotation,
        sit or stand. Accepted requests run the existing trial code and end through the
        existing stop paths.
        """
        cmds = Commands()
        if not isinstance(req, dict):
            return self._skill_reject(cmds, req, 'malformed')
        goal_id, request_id = req.get('goal_id'), req.get('request_id')
        skill, seq, args = req.get('skill'), req.get('seq'), req.get('args', {})
        if (req.get('v') != SKILL_API_VERSION
                or not _valid_id(goal_id) or not _valid_id(request_id)
                or isinstance(seq, bool) or not isinstance(seq, int) or seq < 1
                or not isinstance(args, dict)):
            return self._skill_reject(cmds, req, 'malformed')
        if skill not in SKILLS:
            return self._skill_reject(cmds, req, 'unknown_skill')
        if request_id in self._seen_request_ids:
            return self._skill_reject(cmds, req, 'duplicate_request_id')
        self._remember(self._seen_request_ids, request_id)
        if goal_id in self._retired_goal_ids:
            return self._skill_reject(cmds, req, 'stale_goal')
        if set(args) != SKILL_ARGS[skill] or not all(_valid_id(v) for v in args.values()):
            return self._skill_reject(cmds, req, 'bad_args')

        if goal_id != self._session_goal:
            if skill == SKILL_CANCEL or self._skill is not None:
                return self._skill_reject(
                    cmds, req, 'not_active' if skill == SKILL_CANCEL else 'not_idle')
            # A new goal: everything the previous goal handed over is void.
            if self._session_goal is not None:
                self._remember(self._retired_goal_ids, self._session_goal)
            self._session_goal = goal_id
            self._session_seq = 0
            self._clear_handoffs()
        if seq <= self._session_seq:
            return self._skill_reject(cmds, req, 'out_of_order')
        self._session_seq = seq

        if skill == SKILL_CANCEL:
            target = args['request_id']
            if self._skill is None or target != self._request_id:
                return self._skill_reject(cmds, req, 'not_active')
            if (self._state == State.SIT_AND_IDENTIFY
                    and self._sit_phase in SKILL_POSTURE_PHASES):
                return self._skill_reject(cmds, req, 'posture_phase')
            cmds.log.append(f'Skill {self._skill} [{target}] cancelled by [{request_id}]')
            self._abort(now, cmds, 'cancelled')
            cmds.skill_results[-1]['data']['cancel_request_id'] = request_id
            cmds.skill_results.append(self._result(
                goal_id, request_id, SKILL_CANCEL, 'succeeded', 'cancelled',
                {'request_id': target}))
            return cmds

        if self._estopped:
            return self._skill_reject(cmds, req, 'estopped')
        if self._state != State.IDLE or self._trial is not None:
            return self._skill_reject(cmds, req, 'not_idle')
        if self._skill_seated and skill in SKILL_MOTION:
            return self._skill_reject(cmds, req, 'seated')
        why = self._handoff_problem(skill, args, now)
        if why is not None:
            return self._skill_reject(cmds, req, why)
        self._start_skill_trial(skill, goal_id, request_id, args, now, cmds)
        return cmds

    def _handoff_problem(self, skill: str, args: dict, now: float) -> Optional[str]:
        """Why the earlier step this request depends on cannot be used, or None."""
        cfg = self.config
        if skill == SKILL_ORIENT:
            loc = self._localization
            if loc is None or loc['id'] != args['localization']:
                return 'no_localization'
            if now - loc['measured_s'] > cfg.direction_max_age_s:
                return 'stale_localization'
            if self._moved_since(loc['measured_s']):
                return 'moved_since_localization'
        elif skill == SKILL_APPROACH:
            acq = self._acquisition
            if acq is None or acq['id'] != args['acquisition']:
                return 'no_acquisition'
            if now - acq['t'] > SKILL_ACQUISITION_MAX_S:
                return 'stale_acquisition'
            if self._moved_since(acq['t']):
                return 'moved_since_acquisition'
            if not self._person_fresh(now):
                return 'caller_not_in_view'
        elif skill == SKILL_SIT:
            arr = self._arrival
            if arr is None or arr['id'] != args['arrival']:
                return 'no_arrival'
            if now - arr['t'] > SKILL_ARRIVAL_VALID_S:
                return 'stale_arrival'
            if self._moved_since(arr['t']):
                return 'moved_since_arrival'
        return None

    def _moved_since(self, t: float) -> bool:
        return self._last_motion_s is not None and self._last_motion_s >= t

    def _clear_handoffs(self) -> None:
        self._localization = None
        self._acquisition = None
        self._arrival = None
        self._lost_bearing = None
        self._turn_gate_valid = False

    def _result(self, goal_id, request_id, skill, status, reason, data) -> dict:
        return {'v': SKILL_API_VERSION, 'goal_id': goal_id, 'request_id': request_id,
                'skill': skill, 'status': status, 'reason': reason, 'data': data}

    def _skill_reject(self, cmds: Commands, req, reason: str) -> Commands:
        req = req if isinstance(req, dict) else {}
        ids = [v if _valid_id(v) else None for v in (req.get('goal_id'), req.get('request_id'))]
        skill = req.get('skill') if isinstance(req.get('skill'), str) else None
        cmds.skill_results.append(self._result(ids[0], ids[1], skill, 'rejected', reason, {}))
        cmds.log.append(f'Skill request [{ids[1]}] {skill} rejected: {reason}')
        return cmds

    @staticmethod
    def _remember(memory: List[str], value: str) -> None:
        memory.append(value)
        if len(memory) > GOAL_ID_MEMORY:
            del memory[0]

    def _start_skill_trial(self, skill: str, goal_id: str, request_id: str, args: dict,
                           now: float, cmds: Commands) -> None:
        # Same reset as on_wake (kept as a copy so the baseline entry is untouched).
        self._trial = TrialStats(wake_time_s=now, wake_phrase=f'skill:{skill}',
                                 goal_id=goal_id, skill=skill, request_id=request_id)
        if skill != SKILL_APPROACH:                # the approach keeps the acquired track
            self._reset_person_tracking()
        self._approach_start = None
        self._walk_distance_m = 0.0
        self._wake_s = None                   # no wake phrase is spoken, no hold
        self._turn_sent = False
        self._turn_result = None
        self._turn_done_s = None
        self._last_face = None
        self._turn_purpose = 'doa'
        self._align_turns = 0
        self._search_sign = 0.0               # no search turns inside a skill
        self._search_turns = 0
        self._relistens = 0
        self._relisten_s = None
        self._skill = skill
        self._goal_id = goal_id
        self._request_id = request_id
        self._skill_data = {}
        # An explicit stop first, as on a wake: re-arms the bridge after an e-stop release.
        self._command(cmds, now, 0.0, 0.0)
        cmds.log.append(f'Skill request [{request_id}] {skill} accepted (goal {goal_id})')
        keep_turn_gate = self._turn_gate_valid
        self._turn_gate_valid = False
        if skill == SKILL_LOCALIZE:
            self._localization = None
            self._enter(State.LISTENING, now, cmds, f'skill {skill}')
        elif skill == SKILL_ORIENT:
            loc, self._localization = self._localization, None     # used once
            self._acquisition = self._arrival = None
            self._skill_turn_rad = loc['az']
            self._skill_turn_conf = loc['conf']
            if abs(loc['az']) < self.config.turn_min_rad:
                self._gate_center = loc['az']
                self._trial.gate_center_rad = self._gate_center
                self._skill_done(now, cmds, 'turn_not_needed')
                return
            if self.config.direction_speak_text:
                cmds.say = self.config.direction_speak_text
            self._enter(State.TURN_TO_SOUND, now, cmds, f'skill {skill}')
        elif skill == SKILL_ACQUIRE:
            self._acquisition = self._arrival = None
            # Where the caller should be in the camera: the residual of an immediately
            # preceding turn, else the bearing where a track was just lost, else ahead.
            lost = self._lost_bearing
            if keep_turn_gate:
                center = self._gate_center
            elif lost is not None and now - lost[1] <= self.config.search_timeout_s:
                center = lost[0]
            else:
                center = 0.0
            self._lost_bearing = None
            self._gate_center = center
            self._trial.gate_center_rad = center
            self._enter(State.ACQUIRE_PERSON, now, cmds, f'skill {skill}')
        elif skill == SKILL_APPROACH:
            acq, self._acquisition = self._acquisition, None       # used once
            self._arrival = None
            self._gate_center = acq['bearing'] if self._ema_bearing is None \
                else self._ema_bearing
            self._trial.gate_center_rad = self._gate_center
            self._enter(State.ACQUIRE_PERSON, now, cmds, f'skill {skill}')
        elif skill == SKILL_SIT:
            self._arrival = None                                   # used once
            self._enter(State.SIT_AND_IDENTIFY, now, cmds, f'skill {skill}')
            self._face_received = False
            self._set_sit_phase('settle', now)                     # no final align: no motion
        elif skill == SKILL_ASK_AGAIN:
            self._localization = None
            # Only an utterance after this prompt may localize the caller.
            self._bearing_floor_s = now
            self._asked_s = now
            cmds.say = self.config.relisten_speak_text or 'Please call me again.'
            self._skill_done(now, cmds, 'asked', {'prompt': cmds.say})

    def _skill_done(self, now: float, cmds: Commands, reason: str,
                    data: Optional[dict] = None) -> None:
        """A skill ends here, stopped: trial closed, result emitted."""
        self._command(cmds, now, 0.0, 0.0)
        if data:
            self._skill_data.update(data)
        if self._trial is not None:
            self._trial.stop_reason = reason
        self._finish(now, cmds, reason)

    def _skill_finish_result(self, summary: dict, reason: str, now: float) -> dict:
        """Result for the active skill from its trial summary; clears the skill and
        records the handoff a later step may name."""
        skill, goal_id, request_id = self._skill, self._goal_id, self._request_id
        stop = summary.get('stop_reason') or reason
        data = dict(getattr(self, '_skill_data', {}) or {})
        if summary.get('estop') or reason in ('estop', 'cancelled') or stop in (
                'estop', 'cancelled'):
            # Cancelled even after an arrival: the requested step did not complete.
            status = 'cancelled'
            stop = 'estop' if (summary.get('estop') or 'estop' in (reason, stop)) \
                else 'cancelled'
        elif skill == SKILL_ORIENT:
            ok = stop == 'turn_not_needed' or (
                stop == 'turn_complete'
                and summary.get('turn_result_reason') in SKILL_TURN_OK)
            status = 'succeeded' if ok else 'failed'
        elif skill == SKILL_APPROACH:
            # Same arrival rule as the baseline, walk budget included; the result says
            # which one it was so a supervisor can weigh the evidence.
            status = 'succeeded' if summary.get('success') else 'failed'
        else:
            ok_reason = {SKILL_LOCALIZE: 'localized', SKILL_ACQUIRE: 'acquired',
                         SKILL_SIT: 'sit_commanded', SKILL_ASK_AGAIN: 'asked'}[skill]
            status = 'succeeded' if stop == ok_reason else 'failed'
        if skill == SKILL_ORIENT:
            data.update({k: summary.get(k) for k in (
                'turn_rad', 'turn_turned_rad', 'turn_result_reason', 'gate_center_rad',
                'turn_confidence')})
            if stop == 'turn_not_needed':
                data['turn_result_reason'] = 'turn_not_needed'
        elif skill == SKILL_APPROACH:
            data.update({k: summary.get(k) for k in (
                'final_distance_m', 'final_distance_source', 'final_bearing_rad',
                'final_bbox_h_frac', 'commanded_walk_distance_m', 'align_phases',
                'walk_phases', 'lost_events', 'stale_events')})
            data['stop_reason'] = summary.get('stop_reason')
            data['proximity_evidence'] = stop in ('arrived_bbox', 'arrived_distance')
            data['posture'] = 'standing' if status == 'succeeded' else 'unknown'
        elif skill == SKILL_SIT:
            data['posture'] = 'unverified'      # nothing here reads posture
        if status == 'succeeded':
            if skill == SKILL_LOCALIZE:
                self._localization = dict(self._pending_localization, id=request_id)
            elif skill == SKILL_ACQUIRE:
                self._acquisition = {'id': request_id, 't': now,
                                     'bearing': data['bearing_rad'],
                                     'confidence': data['confidence']}
            elif skill == SKILL_APPROACH:
                self._arrival = {'id': request_id, 't': now}
        self._skill = None
        self._goal_id = None
        self._request_id = None
        self._skill_data = {}
        return self._result(goal_id, request_id, skill, status, stop, data)

    def on_estop(self, engaged: bool, now: float) -> Commands:
        cmds = Commands()
        if engaged:
            newly_engaged = not self._estopped
            self._estopped = True
            self._turn_gate_valid = False     # the robot may be moved by hand
            if self._session_goal is not None:
                self._clear_handoffs()
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
            if self._skill is not None:
                result = self._skill_finish_result(cmds.trial_summary, 'shutdown', now)
                result.update(status='failed', reason='shutdown')
                cmds.skill_results.append(result)
        self._state = State.IDLE
        return cmds

    def tick(self, now: float) -> Commands:
        cmds = Commands()
        self._integrate_walk(now)
        state = self._state
        if state == State.IDLE:
            acq = self._acquisition
            if acq is not None:
                if now - acq['t'] > SKILL_ACQUISITION_MAX_S or not self._person_fresh(now):
                    self._acquisition = None      # the track broke: a new acquire is needed
                elif self._ema_bearing is not None:
                    # Keep perception on the acquired caller until the approach.
                    cmds.gate = (self._ema_bearing, self.config.track_gate_half_rad)
            return cmds
        if state == State.LISTENING:
            self._tick_listening(now, cmds)
        elif state == State.TURN_TO_SOUND:
            self._tick_turn(now, cmds)
        elif state == State.ACQUIRE_PERSON:
            self._tick_acquire(now, cmds)
        elif state in MOTION_STATES:
            self._tick_motion(now, cmds)
        elif state == State.ARRIVED:
            if now - self._state_since + _TIME_EPS >= self.config.arrival_hold_s:
                self._finish(now, cmds, 'arrival hold complete')
        elif state == State.SIT_AND_IDENTIFY:
            self._tick_sit(now, cmds)
        cmds.gate = self._current_gate()
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

    def _tick_localize(self, now: float, cmds: Commands) -> None:
        """localize_caller: report a voice bearing; never turn.

        Only a bearing measured after the robot last moved (a turn changes what every
        older bearing means), after any ask_caller_again prompt, and within
        direction_max_age_s counts.
        """
        cfg = self.config
        floor = max((t for t in (self._bearing_floor_s, self._last_motion_s)
                     if t is not None), default=None)
        fresh = (self._last_dir_s is not None
                 and now - self._last_dir_s <= cfg.direction_max_age_s
                 and (floor is None or self._last_dir_s > floor))
        data = {'bearing_rad': None, 'bearing_deg': None, 'confidence': None, 'age_s': None,
                'threshold': cfg.direction_confidence_threshold}
        if fresh:
            data.update(bearing_rad=round(self._last_azimuth, 4),
                        bearing_deg=round(math.degrees(self._last_azimuth), 1),
                        confidence=round(self._last_dir_confidence, 3),
                        age_s=round(now - self._last_dir_s, 3))
        if fresh and self._last_dir_confidence >= cfg.direction_confidence_threshold:
            self._pending_localization = {
                'az': self._last_azimuth, 'conf': self._last_dir_confidence,
                'measured_s': self._last_dir_s, 'goal': self._goal_id}
            cmds.log.append(f'LOCALIZE: voice at {data["bearing_deg"]:+.0f} deg '
                            f'(confidence {data["confidence"]:.2f}, age {data["age_s"]:.2f} s)')
            self._asked_s = None
            self._skill_done(now, cmds, 'localized', data)
            return
        # After a prompt the caller needs time to answer.
        wait = cfg.relisten_timeout_s if self._asked_s is not None else cfg.listening_timeout_s
        if now - self._state_since > wait:
            self._asked_s = None
            self._skill_done(now, cmds, 'low_confidence' if fresh else 'no_direction', data)

    def _tick_listening(self, now: float, cmds: Commands) -> None:
        cfg = self.config
        if self._skill == SKILL_LOCALIZE:
            self._tick_localize(now, cmds)
            return
        if (self._wake_s is not None
                and now - self._wake_s + _TIME_EPS < cfg.wake_speak_hold_s):
            return                                   # the wake phrase plays before any turn
        fresh = (self._last_dir_s is not None
                 and now - self._last_dir_s <= cfg.direction_max_age_s
                 and (self._wake_s is None
                      or self._last_dir_s >= self._wake_s - max(
                          WAKE_DIRECTION_SLACK_S, cfg.direction_max_age_s))
                 # Re-listening: only a bearing from an utterance after the prompt.
                 and (self._relisten_s is None or self._last_dir_s > self._relisten_s))
        az_deg = math.degrees(self._last_azimuth)
        if fresh and self._last_dir_confidence >= cfg.direction_confidence_threshold:
            if cfg.direction_speak_text:
                cmds.say = cfg.direction_speak_text
            cmds.log.append(f'DOA: voice at {az_deg:+.0f} deg '
                            f'(confidence {self._last_dir_confidence:.2f})')
            if abs(self._last_azimuth) < cfg.turn_min_rad:
                self._gate_center = self._last_azimuth
                if self._trial is not None:
                    self._trial.gate_center_rad = self._gate_center
                self._enter(State.ACQUIRE_PERSON, now, cmds,
                            f'sound ahead ({self._last_azimuth:+.2f} rad), no turn')
            else:
                self._turn_sent = False
                self._turn_result = None
                self._turn_purpose = 'doa'
                self._search_sign = 1.0 if self._last_azimuth > 0.0 else -1.0
                self._enter(State.TURN_TO_SOUND, now, cmds, 'direction confident')
        elif now - self._state_since > (cfg.listening_timeout_s if self._relisten_s is None
                                        else cfg.relisten_timeout_s):
            why = ('no bearing for this utterance' if not fresh else
                   f'bearing {az_deg:+.0f} deg confidence {self._last_dir_confidence:.2f} '
                   f'< {cfg.direction_confidence_threshold}')
            if cfg.require_direction:
                cmds.log.append(f'NOT WALKING: no confident voice direction ({why})')
                self._abort(now, cmds, 'no_direction')
            else:
                self._gate_center = 0.0
                self._enter(State.ACQUIRE_PERSON, now, cmds, f'no confident direction ({why})')

    def _tick_turn(self, now: float, cmds: Commands) -> None:
        cfg = self.config
        trial = self._trial
        if not self._turn_sent:
            self._turn_sent = True
            # The bearing can keep updating during the turn (built-in DOA streams);
            # the result is matched against the angle actually commanded. A skill
            # turn uses the bearing of the localization it names, never a newer one.
            orient = self._skill == SKILL_ORIENT and self._turn_purpose == 'doa'
            target = self._skill_turn_rad if orient else self._last_azimuth
            conf = self._skill_turn_conf if orient else self._last_dir_confidence
            self._turn_target = target
            cmds.rotate_rad = self._turn_target
            self._last_motion_s = now
            if trial is not None and self._turn_purpose == 'doa':
                trial.turn_rad = target
                trial.turn_confidence = conf
            cmds.log.append(f'Rotating toward sound: {target:+.2f} rad '
                            f'(confidence {conf:.2f}); waiting for the turn')
            return
        if self._turn_result is None:
            if now - self._state_since > cfg.turn_result_timeout_s:
                cmds.log.append('NOT WALKING: no rotate_result from the bridge')
                self._abort(now, cmds, 'turn_no_result')
            return
        if now - self._turn_done_s + _TIME_EPS < cfg.turn_settle_s:
            self._last_motion_s = now           # still settling from the turn
            return
        target, turned, reason = self._turn_result
        # Timed (no odometry) turns report no angle: assume the commanded turn.
        residual = 0.0 if turned is None else wrap_pi(target - turned)
        self._gate_center = residual
        if trial is not None and self._turn_purpose == 'doa':
            trial.turn_turned_rad = turned
            trial.turn_result_reason = reason
            trial.gate_center_rad = residual
        cmds.log.append(
            f'{self._turn_purpose.upper()}->turn: requested {math.degrees(target):+.0f} deg, turned '
            f'{"n/a" if turned is None else f"{math.degrees(turned):+.0f}"} deg ({reason}); '
            f'caller expected at {math.degrees(residual):+.0f} deg in the camera')
        if self._skill == SKILL_ORIENT and self._turn_purpose == 'doa':
            self._skill_done(now, cmds, 'turn_complete')
            return
        self._reset_person_tracking()
        self._enter(State.ACQUIRE_PERSON, now, cmds, f'turn {reason}')

    def _tick_acquire(self, now: float, cmds: Commands) -> None:
        cfg = self.config
        trial = self._trial
        if (self._consec_hits >= cfg.search_min_consecutive_detections
                and self._person_fresh(now)):
            if trial is not None:
                if trial.acquired_s is None:
                    trial.acquired_s = now
                    trial.acquire_confidence = self._last_valid_obs.confidence
                    trial.acquire_bearing_rad = self._ema_bearing
                    if cfg.acquired_speak_text and self._skill != SKILL_APPROACH:
                        cmds.say = cfg.acquired_speak_text
                    cmds.log.append(
                        f'Caller acquired at {math.degrees(self._ema_bearing):+.0f} deg '
                        f'(gate center {math.degrees(self._gate_center):+.0f} deg, '
                        f'confidence {self._last_valid_obs.confidence:.2f})')
                else:
                    trial.reacquisitions += 1
            if self._skill == SKILL_ACQUIRE:
                self._skill_acquired(now, cmds)
                return
            reason = self._close_enough_reason()
            if reason is not None:
                self._arrive(now, cmds, reason)
            elif abs(self._ema_bearing) < cfg.approach_align_threshold_rad or (
                    cfg.align_by_rotate
                    and abs(self._ema_bearing) < cfg.approach_realign_threshold_rad):
                self._enter(State.WALK, now, cmds, 'person acquired, aligned')
                self._command(cmds, now, cfg.approach_speed, 0.0)
            elif cfg.align_by_rotate:
                self._align_turn(now, cmds, 'person acquired, misaligned')
            else:
                self._enter(State.ALIGN, now, cmds, 'person acquired, misaligned')
                self._command(cmds, now, 0.0, self._yaw_toward(self._ema_bearing))
            return
        can_relisten = (cfg.relisten_speak_text and not cfg.skip_turn_to_sound
                        and self._approach_start is None
                        and self._relistens < cfg.max_relistens
                        and self._skill is None)
        nobody_seen = self._last_valid_s is None or self._last_valid_s < self._acquire_since
        empty_enough = (cfg.relisten_empty_frames > 0
                        and self._empty_frames >= cfg.relisten_empty_frames)
        if (can_relisten and nobody_seen
                and (empty_enough
                     or now - self._acquire_since + _TIME_EPS >= cfg.relisten_after_s)):
            self._relisten(now, cmds)
            return
        if (cfg.search_turn_rad > 0.0 and self._approach_start is None
                and self._search_sign != 0.0
                and self._search_turns < cfg.max_search_turns
                and (self._last_valid_s is None or self._last_valid_s < self._acquire_since)
                and now - self._acquire_since + _TIME_EPS >= cfg.search_turn_after_s):
            self._search_turns += 1
            self._command(cmds, now, 0.0, 0.0)
            self._last_azimuth = self._search_sign * cfg.search_turn_rad
            self._turn_purpose = 'search'
            self._turn_sent = False
            self._turn_result = None
            self._enter(State.TURN_TO_SOUND, now, cmds,
                        f'nobody in view after the turn: search turn {self._search_turns} of '
                        f'{cfg.max_search_turns} toward the voice side')
            return
        if now - self._acquire_since > cfg.search_timeout_s:
            if can_relisten:                         # someone flickered in view, never acquired
                self._relisten(now, cmds)
                return
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
            if (cfg.walk_budget_arrives and self._ema_bearing is not None
                    and abs(self._ema_bearing) <= cfg.approach_realign_threshold_rad):
                self._arrive(now, cmds, 'arrived_walk_budget')
            else:
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
                if cfg.align_by_rotate:
                    self._align_turn(now, cmds, f'realign bearing={bearing:+.2f}')
                    return
                self._enter(State.ALIGN, now, cmds, f'realign bearing={bearing:+.2f}')
                self._command(cmds, now, 0.0, self._yaw_toward(bearing))
            else:
                self._command(cmds, now, cfg.approach_speed, 0.0)

    def _tick_sit(self, now: float, cmds: Commands) -> None:
        """final_align? -> settle -> sit -> look -> (done | speak_hold) -> stand -> IDLE.

        Sit is only ever sent after a zero velocity and pre_sit_settle_s of standing
        still; the bridge adds its own StopMove and 0.5 s before the Sit request.
        """
        cfg = self.config
        phase = self._sit_phase
        elapsed = now - self._sit_step_since + _TIME_EPS
        if phase == 'final_align':
            bearing = self._ema_bearing
            lost = (not self._person_fresh(now) or bearing is None
                    or (self._miss_since is not None
                        and now - self._miss_since + _TIME_EPS >= cfg.lost_debounce_s))
            done = (not lost and (abs(bearing) <= cfg.final_align_rad
                                  or bearing * self._sit_align_sign < 0.0))
            if lost or done or elapsed >= cfg.final_align_timeout_s:
                why = 'lost' if lost else ('aligned' if done else 'timeout')
                self._command(cmds, now, 0.0, 0.0)
                cmds.log.append(f'Final align {why}'
                                + ('' if bearing is None else f' at {math.degrees(bearing):+.0f} deg'))
                self._set_sit_phase('settle', now)
            else:
                self._command(cmds, now, 0.0, self._yaw_toward(bearing))
        elif phase == 'settle' and elapsed >= cfg.pre_sit_settle_s:
            cmds.sit = True
            cmds.log.append('SIT: StopMove sent, settled; sitting')
            self._set_sit_phase('sit', now)
            if self._skill == SKILL_SIT:
                self._skill_seated = True
                self._last_motion_s = now
        elif phase == 'sit' and elapsed >= cfg.sit_settle_s and self._skill == SKILL_SIT:
            # The sit skill ends once Sit was sent and has had time to take effect;
            # whether the robot is seated is for a posture observer to prove.
            if cfg.speak_text:
                cmds.say = cfg.speak_text
            if self._trial is not None:
                self._trial.stop_reason = 'sit_commanded'
                cmds.trial_summary = self._trial.summary(now)
                self._trial = None
                cmds.skill_results.append(self._skill_finish_result(
                    cmds.trial_summary, 'sit_commanded', now))
            cmds.log.append('DONE: sit sent; operator reset stands the robot up')
            self._set_sit_phase('done', now)
        elif phase == 'sit' and elapsed >= cfg.sit_settle_s:
            cmds.face_request = True
            self._set_sit_phase('look', now)
        elif phase == 'look' and (self._face_received or elapsed >= cfg.face_timeout_s):
            if not self._face_received:
                cmds.log.append('LOOK_AT_FACE: no face result in time')
            if cfg.speak_text:
                cmds.say = cfg.speak_text
            if cfg.sit_hold_until_reset:
                if self._trial is not None:
                    cmds.trial_summary = self._trial.summary(now)
                    self._trial = None
                    if self._skill is not None:
                        cmds.skill_results.append(self._skill_finish_result(
                            cmds.trial_summary, 'seated', now))
                cmds.log.append('DONE: sitting; operator reset stands the robot up')
                self._set_sit_phase('done', now)
            else:
                self._set_sit_phase('speak_hold', now)
        elif phase == 'speak_hold' and elapsed >= cfg.speak_hold_s:
            cmds.stand = True
            self._set_sit_phase('stand', now)
        elif phase == 'stand' and elapsed >= cfg.stand_settle_s:
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
            self._empty_frames = 0
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
        if self._skill == SKILL_APPROACH:
            # Stopped, and the step ends: the supervisor decides about a reacquisition.
            if self._ema_bearing is not None:
                self._lost_bearing = (self._ema_bearing, now)
            cmds.log.append(f'NOT WALKING: caller lost ({why})')
            self._abort(now, cmds, 'track_lost')
            return
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
        # A skill approach always ends standing; only the sit skill sits.
        mode = ARRIVAL_STOP if self._skill is not None else self.config.arrival_mode
        if mode == ARRIVAL_SIT_AND_IDENTIFY:
            self._enter(State.SIT_AND_IDENTIFY, now, cmds, reason)
            self._face_received = False
            bearing = self._ema_bearing
            if (self.config.final_align_rad > 0.0 and bearing is not None
                    and abs(bearing) > self.config.final_align_rad):
                self._sit_align_sign = 1.0 if bearing > 0.0 else -1.0
                self._set_sit_phase('final_align', now)
                self._command(cmds, now, 0.0, self._yaw_toward(bearing))
                cmds.log.append(f'Final align onto the caller at {math.degrees(bearing):+.0f} deg')
            else:
                self._set_sit_phase('settle', now)
        else:
            self._enter(State.ARRIVED, now, cmds, reason)
            if self.config.speak_text and self._skill is None:
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
        keep_gate = (self._skill == SKILL_ORIENT
                     and reason in ('turn_complete', 'turn_not_needed'))
        if self._trial is not None:
            cmds.trial_summary = self._trial.summary(now)
            self._trial = None
            if self._skill is not None:
                result = self._skill_finish_result(cmds.trial_summary, reason, now)
                cmds.skill_results.append(result)
                keep_gate = keep_gate and result['status'] == 'succeeded'
        self._skill = None
        self._turn_gate_valid = keep_gate
        self._enter(State.IDLE, now, cmds, reason)

    def _skill_acquired(self, now: float, cmds: Commands) -> None:
        """acquire_caller ends here: the caller is in view, gated; nothing moved."""
        obs = self._last_valid_obs
        data = {
            'bearing_rad': round(self._ema_bearing, 4),
            'bearing_deg': round(math.degrees(self._ema_bearing), 1),
            'confidence': round(obs.confidence, 3),
            'distance_m': round(obs.distance_m, 3),
            'distance_source': DISTANCE_SOURCES.get(int(obs.distance_source), 'unknown'),
            'bbox_h_frac': round(obs.bbox_h_frac, 3),
            'frame_age_s': round(obs.frame_age_s, 3),
            'age_s': round(now - self._last_valid_s, 3),
            'gate_center_rad': round(self._gate_center, 4),
            'close_enough': self._close_enough_reason(),
            # No persistent track ids exist here: the track is the gated EMA bearing,
            # held until the approach that names this acquisition.
            'track': 'gated_ema',
        }
        self._skill_done(now, cmds, 'acquired', data)

    def _align_turn(self, now: float, cmds: Commands, why: str) -> None:
        """Stop, then one closed-loop odometry turn by the caller's camera bearing."""
        self._command(cmds, now, 0.0, 0.0)
        if self._align_turns >= self.config.max_align_turns:
            cmds.log.append(f'NOT WALKING: caller still misaligned after '
                            f'{self._align_turns} align turns ({why})')
            self._abort(now, cmds, 'align_failed')
            return
        self._align_turns += 1
        self._last_azimuth = self._ema_bearing
        self._turn_purpose = 'align'
        self._turn_sent = False
        self._turn_result = None
        self._enter(State.TURN_TO_SOUND, now, cmds,
                    f'{why}: align turn {self._align_turns} of {self.config.max_align_turns}')

    def _relisten(self, now: float, cmds: Commands) -> None:
        """Caller never seen: ask them to call again and localize the new utterance."""
        self._relistens += 1
        if self._trial is not None:
            self._trial.relistens = self._relistens
        self._command(cmds, now, 0.0, 0.0)
        cmds.say = self.config.relisten_speak_text
        self._relisten_s = now
        self._reset_person_tracking()
        self._search_sign = 0.0
        self._search_turns = 0
        self._enter(State.LISTENING, now, cmds,
                    f'nobody seen: asking the caller to speak again '
                    f'({self._relistens} of {self.config.max_relistens})')

    def _set_sit_phase(self, phase: str, now: float) -> None:
        self._sit_phase = phase
        self._sit_step_since = now

    def _current_gate(self) -> Optional[Tuple[float, float]]:
        """Where perception may select the caller; None = no constraint (idle)."""
        cfg = self.config
        state = self._state
        if state in (State.LISTENING, State.TURN_TO_SOUND):
            return (0.0, 0.0)
        if state == State.ACQUIRE_PERSON:
            if self._approach_start is None or self._ema_bearing is None:
                return (self._gate_center, cfg.acquire_gate_half_rad)
            return (self._ema_bearing, cfg.track_gate_half_rad)
        if state in MOTION_STATES or (state == State.SIT_AND_IDENTIFY
                                      and self._sit_phase == 'final_align'):
            center = self._ema_bearing if self._ema_bearing is not None else self._gate_center
            return (center, cfg.track_gate_half_rad)
        return None

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
        if vx != 0.0 or yaw_rate != 0.0:
            self._last_motion_s = now
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
