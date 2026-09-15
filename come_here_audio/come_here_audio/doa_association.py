"""Pick the direction-of-arrival samples that belong to ONE utterance.

Pure Python, standard library only. No pyusb, no numpy, no ROS, so the
selection rule can be tested on any machine while the hardware provider that
feeds it cannot be imported at all without a ReSpeaker on the bus.

WHY A WINDOW ROUND THE SPEECH ENDPOINT, NOT "THE LAST SECOND"
-------------------------------------------------------------
``ReSpeakerDOAProvider.get_latched_direction(window_s)`` measures its window
backwards from *now*, and *now* is whenever Whisper happened to finish. On this
robot inference takes a few hundred milliseconds to a couple of seconds
depending on utterance length, so "the last second" can be entirely AFTER the
handler stopped talking. The DOA register holds its last value, so that window
still returns a number, and the number can be a chair scraping or the operator
walking away, not the person who said "come here".

The detector already reports the monotonic timestamp of the speech endpoint
(``whisper_phrase_detector._segmenter_loop`` computes it from the sample offset
of the last voiced frame, not from inference completion). This module uses that
timestamp: it takes samples from ``[speech_end - pre_s, speech_end + post_s]``,
prefers the ones the firmware VAD marked active, and refuses to guess when
there are too few.

WHAT IT REFUSES
---------------
* No samples in the window at all -> ``None``. Silence is not a direction.
* Fewer than ``min_active_samples`` VAD-active samples -> the fallback tier,
  reported as such with a lower confidence, so a caller can decline it.
* Non-finite or out-of-range angles are dropped before any statistic is
  computed.

CIRCULAR STATISTICS
-------------------
Azimuths wrap. ``median([-3.0, 3.0, -2.9])`` is 3.0, which is on the other side
of the robot from where every sample actually points. Both the median and the
outlier rejection here work on angles rotated so the circular mean sits at zero,
then rotate back, which is the same correction
``ReSpeakerDOAProvider._circular_median`` makes with numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, List, Optional, Sequence, Tuple


#: One polled sample: (monotonic seconds, azimuth radians, VAD active).
DoaSample = Tuple[float, float, bool]


@dataclass(frozen=True)
class DoaSelection:
    """The direction chosen for one utterance, and how it was chosen."""

    azimuth_rad: float
    confidence: float
    #: 'vad_window' (VAD-active samples round the endpoint), 'window_all'
    #: (samples in the window, none VAD-active), or 'stale_latest' (nothing in
    #: the window; the newest sample before it, at low confidence).
    source: str
    #: Samples considered, VAD-active samples, samples that survived outlier
    #: rejection. Published as diagnostics so a bad turn is explainable.
    n_window: int
    n_active: int
    n_used: int
    #: Seconds between the speech endpoint and the newest sample used.
    age_s: float
    #: How many DISTINCT bearings the used samples contained. The XVF-3000 DOA
    #: register HOLDS its last value and the provider polls at 30 Hz, so one
    #: held bearing arrives as many identical samples that a median counts as
    #: many votes. n_used == 20 with n_distinct == 1 is one measurement wearing
    #: twenty hats. Reported rather than silently corrected: collapsing runs
    #: would also flatten a stationary talker, who is the NORMAL case, so the
    #: robot should measure this on hardware before anyone "fixes" it.
    n_distinct: int = 0
    #: Filled by ``mark_held_register``: the register value just before the
    #: utterance (None when no sample preceded it), how many window samples
    #: moved more than HELD_TOLERANCE_RAD away from it (-1 = not checked), and
    #: whether the register never moved at all during the utterance.
    held_rad: Optional[float] = None
    n_changed: int = -1
    held_register: bool = False


def is_valid_azimuth(value) -> bool:
    """True when ``value`` is a usable bearing: finite and inside +/- pi."""
    try:
        azimuth = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(azimuth) and abs(azimuth) <= math.pi + 1e-9


def wrap_pi(angle: float) -> float:
    """Wrap an angle into [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def circular_mean(angles: Sequence[float]) -> float:
    """Mean direction of a set of angles, wrapped to [-pi, pi]."""
    sin_sum = sum(math.sin(a) for a in angles)
    cos_sum = sum(math.cos(a) for a in angles)
    return math.atan2(sin_sum / len(angles), cos_sum / len(angles))


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def circular_median(angles: Sequence[float]) -> float:
    """Median direction, computed round the circular mean so wrap is handled."""
    reference = circular_mean(angles)
    shifted = [wrap_pi(a - reference) for a in angles]
    return wrap_pi(reference + _median(shifted))


def _percentile(ordered: List[float], fraction: float) -> float:
    """Linear-interpolation percentile over an already sorted list."""
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def reject_outliers(angles: Sequence[float], factor: float = 1.5) -> List[float]:
    """Drop angles outside the IQR fence, measured round the circular mean."""
    values = list(angles)
    if len(values) < 4:
        return values
    reference = circular_mean(values)
    shifted = [wrap_pi(a - reference) for a in values]
    ordered = sorted(shifted)
    q1 = _percentile(ordered, 0.25)
    q3 = _percentile(ordered, 0.75)
    iqr = q3 - q1
    low, high = q1 - factor * iqr, q3 + factor * iqr
    kept = [a for a, s in zip(values, shifted) if low <= s <= high]
    return kept if kept else values


def _count_distinct(angles: Sequence[float], tol: float = 1e-6) -> int:
    """How many genuinely different bearings are in this set.

    The DOA register is 1-degree quantised and HOLDS between voice events, so
    identical floats mean "the array did not update", not "the array measured
    the same thing twice independently".
    """
    distinct: List[float] = []
    for angle in angles:
        if not any(abs(wrap_pi(angle - seen)) <= tol for seen in distinct):
            distinct.append(angle)
    return len(distinct)


def select_direction(
    samples: Iterable[DoaSample],
    *,
    speech_end_s: float,
    speech_start_s: Optional[float] = None,
    pre_s: float = 1.0,
    post_s: float = 0.3,
    min_active_samples: int = 3,
    stale_fallback_s: float = 0.0,
) -> Optional[DoaSelection]:
    """Choose the bearing that belongs to the utterance ending at ``speech_end_s``.

    Parameters
    ----------
    samples:
        ``(monotonic_s, azimuth_rad, vad_active)`` triples, any order.
    speech_end_s:
        Monotonic timestamp of the END of the speech, from the phrase detector.
    speech_start_s:
        Monotonic timestamp of the START of the same speech, from the same
        detector. When given it replaces ``pre_s`` as the window's lower bound
        AND becomes the activity test, so the DOA is trusted over exactly the
        interval ODIN's own adaptive gate accepted as speech. None keeps the
        old fixed-margin, firmware-VAD-only behaviour.
    pre_s, post_s:
        Window round that endpoint. ``pre_s`` covers the utterance itself,
        ``post_s`` the register's settling tail.
    min_active_samples:
        VAD-active samples required before the VAD tier is trusted.
    stale_fallback_s:
        How far BEFORE the window a single last-resort sample may be taken
        from. 0.0 disables the stale tier, which is the safe default: a caller
        that cannot get a fresh direction should search visually rather than
        turn toward a bearing measured before the handler spoke.

    Returns
    -------
    ``DoaSelection`` or ``None`` when nothing usable is in range. ``None`` is a
    real answer and must never be turned into 0.0 rad, which is a command to
    look straight ahead.
    """
    if not math.isfinite(speech_end_s):
        return None
    for name, value in (('pre_s', pre_s), ('post_s', post_s)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f'{name} must be finite and non-negative')

    clean = [
        (float(t), float(az), bool(vad))
        for t, az, vad in samples
        if math.isfinite(t) and is_valid_azimuth(az)
    ]
    if not clean:
        return None

    # The interval the utterance ACTUALLY occupied, when the detector knows
    # it. `pre_s` is a fixed 1.0 s guess that does not shrink for a 0.4 s
    # command, so up to 0.6 s of unrelated room precedes the speech inside the
    # window.
    #
    # Be precise about what that costs, because it is NOT the obvious thing. A
    # register that is merely HOLDING between voice events reports
    # VOICEACTIVITY=0, so tier 1 (`active`) already filters passive hold out.
    # The samples that do damage are from a DIFFERENT acoustic event that
    # genuinely raised the firmware VAD inside the margin: a door, a chair, a
    # cough, someone else. Those arrive flagged active, at a bearing that is
    # not this talker's, and tier 1 cannot tell them apart from the command.
    # Bounding the window by the utterance removes them by construction.
    known_span = (speech_start_s is not None
                  and math.isfinite(speech_start_s)
                  and speech_start_s <= speech_end_s)
    start_s = speech_start_s if known_span else speech_end_s - pre_s
    end_s = speech_end_s + post_s
    window = [s for s in clean if start_s <= s[0] <= end_s]
    active = [s for s in window if s[2]]
    # Samples inside the utterance ODIN's own gate accepted.
    #
    # Two honest caveats, so nobody reads more into this set than it holds:
    #   * It is the utterance HULL (first voiced frame to last), so an internal
    #     pause is inside it. It is not the exact set of frames the gate scored
    #     as voiced; it is the span those frames bracket.
    #   * GAMMAVAD_SR is not left at its default. voice_command_node applies
    #     FAR_FIELD_PROFILE at startup, which lowers it 3.5 -> 2.0 dB, so the
    #     firmware VAD has already been made as sensitive as its registers
    #     allow.
    # What remains is that GAMMAVAD_SR is still an ABSOLUTE threshold while the
    # audio gate is now relative to the measured noise floor, so the two stop
    # agreeing at distance. When they disagree the old code had only the
    # uncorroborated 0.45 tier to offer, which is a bearing the caller is
    # invited to refuse; this tier says instead that the samples fall inside an
    # utterance ODIN itself accepted a command from.
    gated = ([s for s in window if speech_start_s <= s[0] <= speech_end_s]
             if known_span else [])

    if len(active) >= max(1, int(min_active_samples)):
        used = reject_outliers([az for _, az, _ in active])
        azimuth = circular_median(used)
        newest = max(t for t, _, _ in active)
        return DoaSelection(
            azimuth_rad=azimuth,
            confidence=min(0.95, 0.6 + 0.05 * len(used)),
            source='vad_window',
            n_window=len(window),
            n_active=len(active),
            n_used=len(used),
            age_s=speech_end_s - newest,
            n_distinct=_count_distinct(used),
        )

    # The firmware VAD did not latch, but ODIN's own gate did, and the gate is
    # the one that still works at distance. These samples are inside the
    # utterance the robot actually accepted a command from, so they are about
    # this talker; they are simply not corroborated by the array's own fixed
    # threshold. Confidence sits below the corroborated tier and well above the
    # uncorroborated one.
    if len(gated) >= max(1, int(min_active_samples)):
        used = reject_outliers([az for _, az, _ in gated])
        newest = max(t for t, _, _ in gated)
        return DoaSelection(
            azimuth_rad=circular_median(used),
            confidence=min(0.85, 0.5 + 0.05 * len(used)),
            source='gate_window',
            n_window=len(window),
            n_active=len(active),
            n_used=len(used),
            age_s=speech_end_s - newest,
            n_distinct=_count_distinct(used),
        )

    if window:
        # The window covers the utterance but the firmware VAD never latched.
        # That is the documented behaviour while the robot's motors run, so the
        # samples are still about the right moment in time; they are just not
        # corroborated. Report them at a confidence a caller can refuse.
        used = reject_outliers([az for _, az, _ in window])
        newest = max(t for t, _, _ in window)
        return DoaSelection(
            azimuth_rad=circular_median(used),
            confidence=0.45,
            source='window_all',
            n_window=len(window),
            n_active=len(active),
            n_used=len(used),
            age_s=speech_end_s - newest,
            n_distinct=_count_distinct(used),
        )

    if stale_fallback_s > 0.0:
        older = [s for s in clean if start_s - stale_fallback_s <= s[0] < start_s]
        if older:
            newest_t, newest_az, _ = max(older, key=lambda s: s[0])
            return DoaSelection(
                azimuth_rad=newest_az,
                confidence=0.25,
                source='stale_latest',
                n_window=0,
                n_active=0,
                n_used=1,
                age_s=speech_end_s - newest_t,
                n_distinct=1,
            )

    return None


#: A window sample within this of the pre-speech register value did not move.
HELD_TOLERANCE_RAD = math.radians(10.0)
#: Confidence a held bearing is capped to when rejection is on: below the
#: behaviour's direction_confidence_threshold (0.4), so the robot does not turn.
HELD_CONFIDENCE = 0.3


def mark_held_register(
    selection: Optional[DoaSelection],
    samples: Iterable[DoaSample],
    *,
    speech_end_s: float,
    speech_start_s: Optional[float] = None,
    pre_s: float = 1.0,
    post_s: float = 0.3,
    reject_held: bool = False,
) -> Optional[DoaSelection]:
    """Flag a bearing the array never re-aimed for during the utterance.

    Lab 2026-09-15: caller at the robot's right (expected about -90 deg); the
    register already held +163 deg from a noise source behind-left, the
    firmware VAD fired on that noise 6.5 % of the idle time, and the wake got
    +163 deg at confidence 0.95. If no sample inside the utterance moved away
    from the value held before it, the bearing describes whatever the array
    heard LAST, not necessarily this talker.

    A talker standing where the previous sound came from also leaves the
    register unmoved, so rejection is opt-in (``reject_held``); the flag is
    always filled in so lab logs show how often it happens.
    """
    if selection is None or selection.source == 'stale_latest':
        return selection
    clean = sorted(
        (float(t), float(az)) for t, az, _ in samples
        if math.isfinite(t) and is_valid_azimuth(az)
    )
    known_span = (speech_start_s is not None
                  and math.isfinite(speech_start_s)
                  and speech_start_s <= speech_end_s)
    start_s = speech_start_s if known_span else speech_end_s - pre_s
    end_s = speech_end_s + post_s
    before = [az for t, az in clean if t < start_s]
    if not before:
        return replace(selection, held_rad=None, n_changed=-1, held_register=False)
    held = before[-1]
    window = [az for t, az in clean if start_s <= t <= end_s]
    n_changed = sum(1 for az in window if abs(wrap_pi(az - held)) > HELD_TOLERANCE_RAD)
    is_held = bool(window) and n_changed == 0
    confidence = selection.confidence
    if is_held and reject_held:
        confidence = min(confidence, HELD_CONFIDENCE)
    return replace(selection, confidence=confidence, held_rad=held,
                   n_changed=n_changed, held_register=is_held)
