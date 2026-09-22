"""Come Here ANY state machine: the legacy FSM with native-avoidance pursuit (pure Python).

``ComeHereAnyFsm`` subclasses ``ComeHereFsm`` and reuses, unchanged, the whole
caller-intent pipeline: wake, voice direction, turn, relisten, search turns,
DOA-gated acquisition, EMA bearing, lost/stale detection, e-stop, trial stats.
It replaces only the parts the native backend changes:

Pursuit (ALIGN / WALK states), re-evaluated every tick from the latest caller
geometry, never a precomputed path:
  * ``any_control_law: split`` (default): the legacy single-axis law (yaw-only
    ALIGN, forward-only WALK, odometry align turns when ``align_by_rotate``),
    but the forward walk goes through the native avoidance backend, WALK
    re-aligns at once when the caller nears the camera edge
    (``any_fov_stop_rad``), and speed scales down while predicting.
  * ``combined``: forward + yaw together (needs ``any_allow_combined`` here AND
    ``native_allow_combined`` in the bridge, i.e. Stage D evidence).
  * ``vector``: body-frame ``vx = g cos(b)``, ``vy = g sin(b)`` plus yaw that keeps
    the caller in the camera (needs ``any_allow_lateral``, Stage D evidence).
  Commands are ``(vx, vy, yaw_rate)`` on /come_here/cmd_velocity.

Brief occlusion: a short-lived caller estimate in the odometry frame
(``caller_estimate.py``) keeps steering for at most ``any_prediction_ttl_s`` at
reduced speed, only while the camera itself is fresh (empty frames, not a dead
camera). Expiry or growing uncertainty stops and reacquires; the reacquisition
gate is centered on the predicted bearing, and an observation that would put
the caller more than ``any_identity_max_jump_m`` from the estimate is rejected
as a different person. A prediction never produces an arrival.

Bounds (each stops and aborts with its own stop_reason): trial timeout, the
legacy approach timeout and commanded-walk budget (never an arrival here),
odometric travel, displacement from the start, no progress toward the caller,
stale or jumping odometry, stale camera, reacquire timeout (``caller_lost``).

Arrival and final facing: only a fresh visual observation arrives (bbox or
distance), then the robot yaws until the caller's image bearing (the YOLO box
center) is inside ``final_align_rad`` and then VERIFIES it on new frames while
stopped. Only a verified alignment sets the arrival stop_reason and sits.
Lost caller, timeout or too many overshoots: stop, no sit,
stop_reason ``final_align_failed``. Nothing here uses the caller's face, gaze,
head or body orientation: only robot -> caller bearing.
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Optional

from come_here_behavior.caller_estimate import CallerEstimate, Pose2D
from come_here_behavior.come_here_fsm import (
    ARRIVAL_SIT_AND_IDENTIFY,
    ARRIVED_REASONS,
    MOTION_STATES,
    _TIME_EPS,
    ComeHereFsm,
    Commands,
    FsmConfig,
    PersonObservation,
    State,
    TrialStats,
)

CONTROL_LAWS = ('split', 'combined', 'vector')
FINAL_ALIGN_PHASES = ('final_align', 'align_verify')


@dataclass(frozen=True)
class AnyFsmConfig(FsmConfig):
    """FsmConfig plus the ANY fields; field names are behavior_node parameter names."""

    arrival_mode: str = ARRIVAL_SIT_AND_IDENTIFY
    final_align_rad: float = 0.15
    final_align_timeout_s: float = 4.0
    # Control law and the native capabilities it may use (Stage D gates these).
    any_control_law: str = 'split'
    any_allow_combined: bool = False
    any_allow_lateral: bool = False
    any_max_vy: float = 0.3
    any_yaw_gain: float = 1.2
    any_min_speed: float = 0.5          # below this forward speed, hold instead of creeping
    any_fov_stop_rad: float = 0.6       # caller this far off-center: no translation, turn
    # Short-lived caller estimate (brief occlusion).
    any_prediction_ttl_s: float = 1.5
    any_prediction_base_sigma_m: float = 0.25
    any_prediction_sigma_growth_mps: float = 0.8
    any_prediction_max_sigma_m: float = 1.2
    any_camera_max_frame_age_s: float = 1.0
    any_identity_max_jump_m: float = 1.2
    any_identity_memory_s: float = 15.0  # must outlast the reacquire window
    # Odometry and travel bounds.
    any_odom_max_age_s: float = 0.5
    any_odom_max_step_m: float = 0.3
    any_max_travel_m: float = 5.0
    any_max_displacement_m: float = 4.0
    # Progress toward the caller.
    any_no_progress_s: float = 8.0
    any_min_progress_m: float = 0.3
    any_min_progress_bbox: float = 0.05
    any_trial_timeout_s: float = 60.0
    # Final facing.
    any_final_align_yaw: float = 0.6
    any_align_verify_s: float = 0.5
    any_align_verify_obs: int = 2
    any_final_align_attempts: int = 3

    def validate(self) -> None:
        super().validate()
        for name in ('any_max_vy', 'any_yaw_gain', 'any_fov_stop_rad', 'any_prediction_ttl_s',
                     'any_prediction_base_sigma_m', 'any_prediction_max_sigma_m',
                     'any_camera_max_frame_age_s', 'any_identity_max_jump_m',
                     'any_identity_memory_s', 'any_odom_max_age_s', 'any_odom_max_step_m',
                     'any_max_travel_m', 'any_max_displacement_m', 'any_no_progress_s',
                     'any_min_progress_m', 'any_min_progress_bbox', 'any_trial_timeout_s',
                     'any_final_align_yaw', 'final_align_rad', 'final_align_timeout_s'):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f'{name} must be finite and > 0, got {value}')
        for name in ('any_min_speed', 'any_prediction_sigma_growth_mps', 'any_align_verify_s'):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f'{name} must be finite and >= 0, got {value}')
        if self.any_control_law not in CONTROL_LAWS:
            raise ValueError(f'any_control_law must be one of {CONTROL_LAWS}')
        if self.any_control_law == 'combined' and not self.any_allow_combined:
            raise ValueError('any_control_law combined needs any_allow_combined (Stage D)')
        if self.any_control_law == 'vector' and not self.any_allow_lateral:
            raise ValueError('any_control_law vector needs any_allow_lateral (Stage D)')
        if self.walk_budget_arrives:
            raise ValueError('walk_budget_arrives must be false in ANY: path length is not '
                             'arrival when avoidance detours')
        if self.arrival_mode != ARRIVAL_SIT_AND_IDENTIFY:
            raise ValueError('ANY needs arrival_mode sit_and_identify (verified final facing)')
        if self.any_align_verify_obs < 1 or self.any_final_align_attempts < 1:
            raise ValueError('any_align_verify_obs and any_final_align_attempts must be >= 1')
        if self.any_identity_memory_s < self.search_timeout_s:
            raise ValueError('any_identity_memory_s must be >= search_timeout_s: otherwise a '
                             'long reacquisition accepts anyone as the caller')
        if self.any_min_speed > self.approach_speed:
            raise ValueError('any_min_speed must be <= approach_speed')


@dataclass
class AnyTrialStats(TrialStats):
    control_law: str = 'split'
    travel_distance_m: float = 0.0
    max_displacement_m: float = 0.0
    initial_range_m: Optional[float] = None
    last_range_m: Optional[float] = None
    caller_prediction_used_s: float = 0.0
    caller_loss_events: int = 0
    caller_prediction_expired_events: int = 0
    identity_rejections: int = 0
    no_progress_events: int = 0
    camera_stale_events: int = 0
    lateral_commands: int = 0
    final_align_start_bearing_rad: Optional[float] = None
    final_align_end_bearing_rad: Optional[float] = None
    final_align_verified: Optional[bool] = None
    final_align_attempts: int = 0
    final_align_failure: Optional[str] = None

    def summary(self, end_s: float) -> dict:
        def rnd(v):
            return None if v is None else round(v, 3)

        d = super().summary(end_s)
        progress = None
        if self.initial_range_m is not None and self.last_range_m is not None:
            progress = self.initial_range_m - self.last_range_m
        d.update({
            'mode': 'any',
            'control_law': self.control_law,
            'travel_distance_m': rnd(self.travel_distance_m),
            'max_displacement_m': rnd(self.max_displacement_m),
            'straight_line_progress_m': rnd(progress),
            'caller_prediction_used_s': rnd(self.caller_prediction_used_s),
            'caller_reacquisitions': self.reacquisitions,
            'caller_loss_events': self.caller_loss_events,
            'caller_prediction_expired_events': self.caller_prediction_expired_events,
            'identity_rejections': self.identity_rejections,
            'no_progress_events': self.no_progress_events,
            'camera_stale_events': self.camera_stale_events,
            'lateral_commands': self.lateral_commands,
            'final_align_start_bearing_rad': rnd(self.final_align_start_bearing_rad),
            'final_align_end_bearing_rad': rnd(self.final_align_end_bearing_rad),
            'final_align_verified': self.final_align_verified,
            'final_align_attempts': self.final_align_attempts,
            'final_align_failure': self.final_align_failure,
            'success': (self.stop_reason in ARRIVED_REASONS and not self.estop
                        and self.final_align_verified is True),
        })
        return d


class ComeHereAnyFsm(ComeHereFsm):
    def __init__(self, config: AnyFsmConfig):
        if not isinstance(config, AnyFsmConfig):
            raise TypeError('ComeHereAnyFsm needs an AnyFsmConfig')
        super().__init__(config)
        self._caller = CallerEstimate(
            ttl_s=config.any_prediction_ttl_s,
            base_sigma_m=config.any_prediction_base_sigma_m,
            sigma_growth_mps=config.any_prediction_sigma_growth_mps,
            max_sigma_m=config.any_prediction_max_sigma_m)
        self._pose: Optional[Pose2D] = None
        self._pose_s: Optional[float] = None
        self._pos_obs_count = 0
        self._last_frame_age = math.inf
        self._last_tick_s = 0.0
        self._reset_any()

    def _reset_any(self) -> None:
        self._start_pose: Optional[Pose2D] = None
        self._travel_m = 0.0
        self._odom_invalid: Optional[str] = None
        self._best_range: Optional[float] = None
        self._best_bbox: Optional[float] = None
        self._progress_s: Optional[float] = None
        self._predicting_since: Optional[float] = None
        self._pending_arrival: Optional[str] = None
        self._align_started_s = 0.0
        self._verify_since = 0.0
        self._verify_obs_start = 0

    # -- inputs --

    def on_wake(self, phrase: str, now: float) -> Commands:
        had_trial = self._trial is not None
        cmds = super().on_wake(phrase, now)
        if self._trial is not None and not had_trial:
            self._trial = AnyTrialStats(**dataclasses.asdict(self._trial),
                                        control_law=self.config.any_control_law)
            self._reset_any()
            self._caller.reset()
            self._start_pose = self._pose if self._odom_fresh(now) else None
        return cmds

    def on_odom(self, x: float, y: float, yaw: float, now: float) -> Commands:
        cmds = Commands()
        pose = Pose2D(float(x), float(y), float(yaw))
        if not pose.finite():
            cmds.log.append('Ignoring non-finite odometry')
            return cmds
        prev = self._pose
        self._pose, self._pose_s = pose, now
        trial = self._trial
        if trial is None or not isinstance(trial, AnyTrialStats):
            return cmds
        if self._start_pose is None:
            self._start_pose = pose
        if prev is not None:
            step = math.hypot(pose.x - prev.x, pose.y - prev.y)
            if step > self.config.any_odom_max_step_m:
                self._odom_invalid = f'odometry jumped {step:.2f} m in one sample'
            else:
                self._travel_m += step
                trial.travel_distance_m = self._travel_m
        trial.max_displacement_m = max(trial.max_displacement_m, self._displacement())
        return cmds

    def on_person(self, obs: Optional[PersonObservation], now: float) -> Commands:
        cfg = self.config
        trial = self._trial
        identity_log = None
        if obs is not None and all(math.isfinite(v) for v in (obs.frame_age_s,)):
            self._last_frame_age = obs.frame_age_s
        else:
            self._last_frame_age = math.inf
        if self.classify(obs) == 'positive' and self._approach_start is not None:
            reject = None
            if not self._odom_fresh(now) and self._caller.has_fix:
                reject = 'odometry stale, cannot check it is the same caller'
            elif not self._caller.consistent(self._pose, obs.bearing_rad, obs.distance_m,
                                             now, cfg.any_identity_max_jump_m,
                                             cfg.any_identity_memory_s):
                reject = 'not where the tracked caller can be'
            if reject is not None:
                if isinstance(trial, AnyTrialStats):
                    trial.identity_rejections += 1
                identity_log = (f'Rejected a person at {math.degrees(obs.bearing_rad):+.0f} deg '
                                f'{obs.distance_m:.2f} m: {reject}')
                obs = dataclasses.replace(obs, detected=0.0, confidence=0.0)
        cmds = super().on_person(obs, now)
        if identity_log:
            cmds.log.append(identity_log)
        if self.classify(obs) == 'positive':
            self._pos_obs_count += 1
            if self._odom_fresh(now):
                self._caller.observe(self._pose, obs.bearing_rad, obs.distance_m, now)
            if isinstance(trial, AnyTrialStats) and self._state in MOTION_STATES:
                self._update_progress(obs, now, trial)
        return cmds

    def _update_progress(self, obs: PersonObservation, now: float, trial: AnyTrialStats) -> None:
        cfg = self.config
        progressed = False
        if obs.distance_m > 0.0:
            if trial.initial_range_m is None:
                trial.initial_range_m = obs.distance_m
            trial.last_range_m = obs.distance_m
            if self._best_range is None:
                self._best_range = obs.distance_m
            elif obs.distance_m <= self._best_range - cfg.any_min_progress_m:
                self._best_range = obs.distance_m
                progressed = True
        if self._best_bbox is None:
            self._best_bbox = obs.bbox_h_frac
        elif obs.bbox_h_frac >= self._best_bbox + cfg.any_min_progress_bbox:
            self._best_bbox = obs.bbox_h_frac
            progressed = True
        if progressed:
            self._progress_s = now

    # -- tick --

    def tick(self, now: float) -> Commands:
        cfg = self.config
        trial = self._trial
        self._last_tick_s = now
        in_trial_phase = (self._state != State.SIT_AND_IDENTIFY
                          or self._sit_phase in FINAL_ALIGN_PHASES)
        if trial is not None and self._state != State.IDLE and in_trial_phase:
            cmds = Commands()
            if self._odom_invalid is not None:
                cmds.log.append(f'STOP: {self._odom_invalid}')
                self._abort(now, cmds, 'odom_invalid')
                cmds.gate = self._current_gate()
                return cmds
            if now - trial.wake_time_s > cfg.any_trial_timeout_s:
                self._abort(now, cmds, 'trial_timeout')
                cmds.gate = self._current_gate()
                return cmds
        return super().tick(now)

    def _tick_motion(self, now: float, cmds: Commands) -> None:
        cfg = self.config
        trial = self._trial
        if not self._odom_fresh(now):
            cmds.log.append('STOP: odometry stale; ANY cannot bound travel without it')
            self._abort(now, cmds, 'odom_stale')
            return
        if self._travel_m >= cfg.any_max_travel_m:
            self._abort(now, cmds, 'travel_budget')
            return
        if self._displacement() >= cfg.any_max_displacement_m:
            self._abort(now, cmds, 'displacement_budget')
            return
        if now - self._approach_start > cfg.approach_timeout_s:
            self._abort(now, cmds, 'approach_timeout')
            return
        if self._walk_distance_m >= cfg.max_walk_distance_m:
            self._abort(now, cmds, 'walk_budget')     # never an arrival in ANY
            return
        if not self._camera_ok(now):
            if trial is not None:
                trial.stale_events += 1
                trial.camera_stale_events += 1
            self._end_prediction(now)
            self._lose(now, cmds, 'camera stale: no fresh frames, stopping')
            return

        visual_lost = (self._miss_since is not None
                       and now - self._miss_since + _TIME_EPS >= cfg.lost_debounce_s)
        if self._person_fresh(now) and not visual_lost:
            self._end_prediction(now)
            reason = self._close_enough_reason()
            if reason is not None:
                self._arrive(now, cmds, reason)
                return
            bearing, scale = self._ema_bearing, 1.0
        else:
            pred = self._caller.predict(self._pose, now)
            if pred is None:
                if trial is not None:
                    trial.lost_events += 1
                    if self._predicting_since is not None:
                        trial.caller_prediction_expired_events += 1
                why = ('caller_prediction_expired: stopping to reacquire'
                       if self._predicting_since is not None
                       else 'caller not visible and no caller estimate: stopping')
                self._end_prediction(now)
                self._lose(now, cmds, why)
                return
            if self._predicting_since is None:
                self._predicting_since = now
                if trial is not None:
                    trial.caller_loss_events += 1
                cmds.log.append(
                    f'Caller occluded: steering on the estimate at '
                    f'{math.degrees(pred.bearing_rad):+.0f} deg {pred.range_m:.2f} m '
                    f'(ttl {cfg.any_prediction_ttl_s:.1f} s, no arrival until seen again)')
            bearing = pred.bearing_rad
            scale = max(0.0, 1.0 - pred.age_s / cfg.any_prediction_ttl_s)

        if self._progress_s is None:
            self._progress_s = self._approach_start
        if now - self._progress_s > cfg.any_no_progress_s:
            if trial is not None:
                trial.no_progress_events += 1
            cmds.log.append(f'STOP: no progress toward the caller in {cfg.any_no_progress_s:.0f} s')
            self._abort(now, cmds, 'no_progress')
            return
        self._drive(now, cmds, bearing, scale)

    def _drive(self, now: float, cmds: Commands, bearing: float, scale: float) -> None:
        cfg = self.config
        law = cfg.any_control_law
        if scale >= 1.0:
            speed = cfg.approach_speed
        elif scale > 0.0:
            # Predicting: slow down toward the gait floor (mcf trots cleanly only at
            # >= any_min_speed), then stop when the estimate expires or grows uncertain.
            speed = max(cfg.any_min_speed, cfg.approach_speed * scale)
        else:
            speed = 0.0
        if law == 'split':
            elapsed = now - self._phase_since
            if self._state == State.ALIGN:
                if (abs(bearing) < cfg.approach_align_threshold_rad
                        and elapsed + _TIME_EPS >= cfg.approach_min_align_s):
                    self._enter(State.WALK, now, cmds, f'aligned bearing={bearing:+.2f}')
                    self._command(cmds, now, speed, 0.0)
                else:
                    self._command(cmds, now, 0.0, self._yaw_toward(bearing))
                return
            off_center = abs(bearing) > cfg.any_fov_stop_rad
            if off_center or (elapsed + _TIME_EPS >= cfg.approach_min_walk_s
                              and abs(bearing) > cfg.approach_realign_threshold_rad):
                why = (f'caller near the camera edge bearing={bearing:+.2f}' if off_center
                       else f'realign bearing={bearing:+.2f}')
                if cfg.align_by_rotate and self._predicting_since is None:
                    self._align_turn(now, cmds, why)
                    return
                self._enter(State.ALIGN, now, cmds, why)
                self._command(cmds, now, 0.0, self._yaw_toward(bearing))
            else:
                self._command(cmds, now, speed, 0.0)
            return
        if self._state == State.ALIGN:
            self._enter(State.WALK, now, cmds, f'{law} pursuit')
        yaw = max(-cfg.approach_cw_yaw, min(cfg.approach_ccw_yaw, cfg.any_yaw_gain * bearing))
        if abs(bearing) > cfg.any_fov_stop_rad:
            self._command(cmds, now, 0.0, yaw)
            return
        if law == 'combined':
            vx = speed * max(0.0, math.cos(bearing))
            self._command(cmds, now, vx if vx >= cfg.any_min_speed else 0.0, yaw)
            return
        vx = speed * max(0.0, math.cos(bearing))
        vy = max(-cfg.any_max_vy, min(cfg.any_max_vy, speed * math.sin(bearing)))
        if math.hypot(vx, vy) < cfg.any_min_speed:
            vx = vy = 0.0
        self._command(cmds, now, vx, yaw, vy)

    # -- arrival and verified final facing --

    def _arrive(self, now: float, cmds: Commands, reason: str) -> None:
        self._command(cmds, now, 0.0, 0.0)
        self._end_prediction(now)
        trial = self._trial
        if trial is not None:
            trial.approach_end_s = now
            if isinstance(trial, AnyTrialStats):
                trial.final_align_start_bearing_rad = self._ema_bearing
        self._pending_arrival = reason
        obs = self._last_valid_obs
        cmds.log.append(f'Close enough ({reason}, fresh camera): bbox_h_frac={obs.bbox_h_frac:.2f} '
                        f'distance={obs.distance_m:.2f} m; facing the caller before sitting')
        self._enter(State.SIT_AND_IDENTIFY, now, cmds, reason)
        self._face_received = False
        self._align_started_s = now
        self._set_sit_phase('final_align', now)

    def _tick_sit(self, now: float, cmds: Commands) -> None:
        if self._sit_phase in FINAL_ALIGN_PHASES:
            self._tick_final_align(now, cmds)
        else:
            super()._tick_sit(now, cmds)

    def _tick_final_align(self, now: float, cmds: Commands) -> None:
        cfg = self.config
        trial = self._trial
        bearing = self._ema_bearing
        obs = self._last_valid_obs
        lost = (not self._person_fresh(now) or bearing is None or obs is None
                or not self._camera_ok(now)
                or (self._miss_since is not None
                    and now - self._miss_since + _TIME_EPS >= cfg.lost_debounce_s))
        if lost:
            self._final_align_fail(now, cmds, 'caller not freshly visible')
            return
        if now - self._align_started_s > cfg.final_align_timeout_s:
            self._final_align_fail(now, cmds, f'not verified within {cfg.final_align_timeout_s:.1f} s '
                                              f'(bearing {math.degrees(bearing):+.0f} deg)')
            return
        tol = cfg.final_align_rad
        aligned = abs(bearing) <= tol and abs(obs.bearing_rad) <= tol
        if self._sit_phase == 'final_align':
            if aligned:
                self._command(cmds, now, 0.0, 0.0)
                self._set_sit_phase('align_verify', now)
                self._verify_since = now
                self._verify_obs_start = self._pos_obs_count
            else:
                yaw = cfg.any_final_align_yaw if bearing > 0.0 else -cfg.any_final_align_yaw
                self._command(cmds, now, 0.0, yaw)
            return
        new_obs = self._pos_obs_count - self._verify_obs_start
        if new_obs > 0 and not aligned:
            attempts = (trial.final_align_attempts + 1
                        if isinstance(trial, AnyTrialStats) else cfg.any_final_align_attempts)
            if isinstance(trial, AnyTrialStats):
                trial.final_align_attempts = attempts
            if attempts >= cfg.any_final_align_attempts:
                self._final_align_fail(now, cmds, f'{attempts} verify attempts drifted off')
                return
            cmds.log.append(f'Final align verify: caller at {math.degrees(bearing):+.0f} deg, '
                            're-aligning')
            self._set_sit_phase('final_align', now)
            yaw = cfg.any_final_align_yaw if bearing > 0.0 else -cfg.any_final_align_yaw
            self._command(cmds, now, 0.0, yaw)
            return
        self._command(cmds, now, 0.0, 0.0)
        if (aligned and new_obs >= cfg.any_align_verify_obs
                and now - self._verify_since + _TIME_EPS >= cfg.any_align_verify_s):
            if trial is not None:
                trial.stop_reason = self._pending_arrival
                if isinstance(trial, AnyTrialStats):
                    trial.final_align_verified = True
                    trial.final_align_end_bearing_rad = bearing
                self._record_final()
            cmds.log.append(f'Final align VERIFIED at {math.degrees(bearing):+.1f} deg on '
                            f'{new_obs} new frames; settling, then sit')
            self._set_sit_phase('settle', now)

    def _final_align_fail(self, now: float, cmds: Commands, why: str) -> None:
        trial = self._trial
        if isinstance(trial, AnyTrialStats):
            trial.final_align_verified = False
            trial.final_align_end_bearing_rad = self._ema_bearing
            trial.final_align_failure = why
        cmds.log.append(f'NOT SITTING: final alignment failed ({why}); stopped')
        self._abort(now, cmds, 'final_align_failed')

    # -- transitions and helpers --

    def _abort(self, now: float, cmds: Commands, reason: str) -> None:
        if reason == 'reacquire_timeout':
            reason = 'caller_lost'
        self._end_prediction(now)
        super()._abort(now, cmds, reason)

    def _current_gate(self):
        cfg = self.config
        state = self._state
        if state == State.SIT_AND_IDENTIFY and self._sit_phase in FINAL_ALIGN_PHASES:
            center = self._ema_bearing if self._ema_bearing is not None else self._gate_center
            return (center, cfg.track_gate_half_rad)
        predicting = (state in MOTION_STATES and self._predicting_since is not None) or (
            state == State.ACQUIRE_PERSON and self._approach_start is not None)
        if predicting and self._pose is not None:
            pred = self._caller.predict(self._pose, self._last_tick_s)
            if pred is not None:
                return (pred.bearing_rad, cfg.track_gate_half_rad)
        return super()._current_gate()

    def _command(self, cmds: Commands, now: float, vx: float, yaw_rate: float,
                 vy: float = 0.0) -> None:
        super()._command(cmds, now, vx, yaw_rate)
        cmds.velocity = (float(vx), float(vy), float(yaw_rate))
        trial = self._trial
        if trial is not None and vy != 0.0:
            if vx == 0.0 and yaw_rate == 0.0:
                trial.motion_commands += 1
            elif vx == 0.0:
                trial.motion_commands += 1
                trial.combined_commands += 1
            if isinstance(trial, AnyTrialStats):
                trial.lateral_commands += 1

    def _odom_fresh(self, now: float) -> bool:
        return (self._pose is not None and self._pose_s is not None
                and now - self._pose_s <= self.config.any_odom_max_age_s + _TIME_EPS)

    def _camera_ok(self, now: float) -> bool:
        return (self._last_person_msg_s is not None
                and now - self._last_person_msg_s <= self.config.person_stale_timeout_s + _TIME_EPS
                and self._last_frame_age <= self.config.any_camera_max_frame_age_s)

    def _displacement(self) -> float:
        if self._start_pose is None or self._pose is None:
            return 0.0
        return math.hypot(self._pose.x - self._start_pose.x, self._pose.y - self._start_pose.y)

    def _end_prediction(self, now: float) -> None:
        if self._predicting_since is not None:
            trial = self._trial
            if isinstance(trial, AnyTrialStats):
                trial.caller_prediction_used_s += max(0.0, now - self._predicting_since)
            self._predicting_since = None

    def viewer_status(self, now: float) -> dict:
        status = super().viewer_status(now)
        status.update({
            'mode': 'any',
            'predicting': self._predicting_since is not None,
            'travel_m': round(self._travel_m, 2),
            'odom_fresh': self._odom_fresh(now),
        })
        return status
