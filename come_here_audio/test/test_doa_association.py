"""Tests for associating direction-of-arrival samples with one utterance.

Pure Python: no ReSpeaker, no pyusb, no numpy. The point of putting the
selection rule in its own module was so this could run anywhere, because the
provider that feeds it cannot even be IMPORTED without a microphone on the bus.
"""
from __future__ import annotations

import math

import pytest

from come_here_audio.doa_association import (
    circular_mean,
    circular_median,
    is_valid_azimuth,
    reject_outliers,
    select_direction,
    wrap_pi,
)


SPEECH_END = 1000.0


def _samples(entries):
    return [(t, az, vad) for t, az, vad in entries]


def test_vad_samples_around_the_speech_end_are_preferred():
    selection = select_direction(_samples([
        (999.4, 0.50, True), (999.6, 0.55, True), (999.8, 0.52, True),
        (1000.9, -2.00, True),   # a second later: after the utterance
    ]), speech_end_s=SPEECH_END)
    assert selection is not None
    assert selection.source == 'vad_window'
    assert selection.azimuth_rad == pytest.approx(0.52, abs=0.03)
    assert selection.n_active == 3


def test_samples_from_after_the_utterance_do_not_decide_the_direction():
    """The failure this module exists to prevent, stated as a test.

    Inference finishes late, and the DOA register holds its last value. A
    "last second from now" window would return -2.0 rad here. Associating with
    the speech endpoint returns the direction the handler actually spoke from.
    """
    now = SPEECH_END + 1.6      # where Whisper finished
    entries = [(999.5, 0.60, True), (999.7, 0.62, True), (999.9, 0.58, True)]
    entries += [(now - 0.9 + 0.1 * i, -2.0, True) for i in range(9)]
    selection = select_direction(_samples(entries), speech_end_s=SPEECH_END)
    assert selection is not None
    assert selection.azimuth_rad == pytest.approx(0.60, abs=0.05)
    assert selection.azimuth_rad > 0.0


def test_no_samples_near_the_endpoint_returns_none_not_zero():
    """None means 'no direction'. 0.0 rad means 'face straight ahead'."""
    selection = select_direction(_samples([
        (900.0, 0.5, True), (901.0, 0.5, True),
    ]), speech_end_s=SPEECH_END)
    assert selection is None


def test_too_few_vad_samples_falls_back_and_says_so():
    """Motor noise pins the firmware VAD to zero; the tier is reported, not hidden."""
    selection = select_direction(_samples([
        (999.5, 0.40, False), (999.7, 0.42, False), (999.9, 0.41, False),
    ]), speech_end_s=SPEECH_END, min_active_samples=3)
    assert selection is not None
    assert selection.source == 'window_all'
    assert selection.n_active == 0
    assert selection.confidence < 0.6
    assert selection.azimuth_rad == pytest.approx(0.41, abs=0.02)


def test_the_stale_tier_is_off_by_default_and_opt_in():
    older = _samples([(996.0, 1.1, True)])
    assert select_direction(older, speech_end_s=SPEECH_END) is None
    opted_in = select_direction(older, speech_end_s=SPEECH_END,
                                stale_fallback_s=10.0)
    assert opted_in is not None
    assert opted_in.source == 'stale_latest'
    assert opted_in.confidence <= 0.3


def test_non_finite_and_out_of_range_samples_are_dropped_before_statistics():
    selection = select_direction(_samples([
        (999.5, float('nan'), True), (999.6, 50.0, True),
        (999.7, 0.30, True), (999.8, 0.32, True), (999.9, 0.31, True),
    ]), speech_end_s=SPEECH_END)
    assert selection is not None
    assert selection.n_active == 3
    assert selection.azimuth_rad == pytest.approx(0.31, abs=0.02)


def test_an_outlier_does_not_drag_the_answer_across_the_room():
    selection = select_direction(_samples([
        (999.5, 0.50, True), (999.6, 0.52, True), (999.7, 0.51, True),
        (999.8, 0.49, True), (999.85, -2.90, True),
    ]), speech_end_s=SPEECH_END)
    assert selection is not None
    assert selection.azimuth_rad == pytest.approx(0.505, abs=0.05)


def test_wraparound_is_handled_rather_than_averaged_through_zero():
    """Samples straddling +/- pi must not resolve to the opposite direction."""
    selection = select_direction(_samples([
        (999.6, 3.10, True), (999.7, -3.10, True), (999.8, 3.13, True),
    ]), speech_end_s=SPEECH_END)
    assert selection is not None
    assert abs(selection.azimuth_rad) > 3.0


def test_a_non_finite_speech_end_is_refused():
    assert select_direction(_samples([(999.9, 0.5, True)]),
                            speech_end_s=float('nan')) is None


def test_negative_window_bounds_are_a_programming_error():
    with pytest.raises(ValueError):
        select_direction([], speech_end_s=SPEECH_END, pre_s=-1.0)


def test_azimuth_validity_matches_the_ros_convention():
    assert is_valid_azimuth(0.0)
    assert is_valid_azimuth(math.pi)
    assert is_valid_azimuth(-math.pi)
    assert not is_valid_azimuth(4.0)
    assert not is_valid_azimuth(float('inf'))
    assert not is_valid_azimuth(float('nan'))
    assert not is_valid_azimuth('left')


def test_circular_helpers_agree_with_the_numpy_implementation_they_replace():
    angles = [3.10, -3.10, 3.13]
    assert abs(wrap_pi(circular_mean(angles))) > 3.0
    assert abs(circular_median(angles)) > 3.0
    assert circular_median([0.1, 0.2, 0.3]) == pytest.approx(0.2, abs=1e-6)
    assert reject_outliers([0.5, 0.51, 0.49, 0.52, -3.0]) == [
        0.5, 0.51, 0.49, 0.52]


# --- The gate tier: DOA as sensitive as the microphone -----------------------
#
# Added 2026-09-10. The adaptive RMS gate (docs/34 section 8c) made ODIN hear
# commands at distances where the array's own firmware VAD, a FIXED dB
# threshold, stops latching. Until these tests the bearing for exactly those
# commands fell to the uncorroborated 0.45 tier, so the robot heard you and
# then did not trust where you were.

SPEECH_START = SPEECH_END - 0.8


def test_the_gate_interval_is_trusted_when_the_firmware_vad_never_latches():
    """The far-field case: ODIN's gate opened, the array's fixed VAD did not."""
    selection = select_direction(_samples([
        (999.4, 0.50, False), (999.6, 0.55, False), (999.8, 0.52, False),
    ]), speech_end_s=SPEECH_END, speech_start_s=SPEECH_START)
    assert selection is not None
    assert selection.source == 'gate_window'
    assert selection.azimuth_rad == pytest.approx(0.52, abs=0.03)
    assert selection.n_active == 0        # the firmware never agreed
    assert selection.confidence > 0.45    # and it is still worth more than 'window_all'


def test_without_the_speech_start_the_same_samples_stay_uncorroborated():
    """The old behaviour is exactly preserved when the span is not supplied."""
    selection = select_direction(_samples([
        (999.4, 0.50, False), (999.6, 0.55, False), (999.8, 0.52, False),
    ]), speech_end_s=SPEECH_END)
    assert selection is not None
    assert selection.source == 'window_all'
    assert selection.confidence == pytest.approx(0.45)


def test_a_held_bearing_from_before_the_utterance_does_not_decide_it():
    """The register HOLDS, so the fixed 1.0 s margin imports a stale bearing.

    A door slams at 999.1 and the array latches -2.0 rad. The handler starts
    speaking at 999.2 from +0.6 rad. With the old fixed pre_s the stale samples
    are inside the window and outvote the real ones; with the real utterance
    span they are simply not part of this utterance.
    """
    entries = [(999.0 + 0.02 * i, -2.0, True) for i in range(9)]   # before speech
    entries += [(999.3, 0.60, False), (999.5, 0.62, False), (999.7, 0.58, False)]
    stale_wins = select_direction(_samples(entries), speech_end_s=SPEECH_END)
    assert stale_wins is not None
    assert stale_wins.azimuth_rad < 0.0          # the door, not the handler

    selection = select_direction(_samples(entries), speech_end_s=SPEECH_END,
                                 speech_start_s=999.2)
    assert selection is not None
    assert selection.source == 'gate_window'
    assert selection.azimuth_rad == pytest.approx(0.60, abs=0.05)


def test_the_firmware_vad_still_wins_when_it_does_latch():
    """Corroboration is better evidence, so the near-field tier is unchanged."""
    selection = select_direction(_samples([
        (999.4, 0.50, True), (999.6, 0.55, True), (999.8, 0.52, True),
    ]), speech_end_s=SPEECH_END, speech_start_s=SPEECH_START)
    assert selection is not None
    assert selection.source == 'vad_window'
    assert selection.confidence > 0.7


def test_a_held_register_is_reported_not_silently_averaged():
    """20 polls of one held value are one measurement, and must be visible.

    Collapsing them is NOT done here: a stationary talker also produces a
    constant bearing, and that is the normal case. The count is reported so a
    lab session can tell the two apart on real hardware.
    """
    entries = [(999.3 + 0.01 * i, 0.60, False) for i in range(20)]
    selection = select_direction(_samples(entries), speech_end_s=SPEECH_END,
                                 speech_start_s=SPEECH_START)
    assert selection is not None
    assert selection.n_used == 20
    assert selection.n_distinct == 1


def test_an_impossible_speech_span_falls_back_instead_of_trusting_it():
    """start after end is a programming error upstream, not a direction."""
    entries = [(999.4, 0.50, False), (999.6, 0.55, False), (999.8, 0.52, False)]
    selection = select_direction(_samples(entries), speech_end_s=SPEECH_END,
                                 speech_start_s=SPEECH_END + 5.0)
    assert selection is not None
    assert selection.source == 'window_all'


def test_a_non_finite_speech_start_is_ignored_not_propagated():
    entries = [(999.4, 0.50, False), (999.6, 0.55, False), (999.8, 0.52, False)]
    selection = select_direction(_samples(entries), speech_end_s=SPEECH_END,
                                 speech_start_s=float('nan'))
    assert selection is not None
    assert selection.source == 'window_all'


def test_control_transfer_timeout_is_milliseconds_not_microseconds():
    """A USB control timeout is in ms, and this one must stay sane.

    pyusb passes this value straight to libusb_control_transfer, whose timeout
    is documented in milliseconds. It was 100000 with a comment claiming
    microseconds, which is a 100 SECOND block per failed read. The direction
    poll loop catches read failures and continues, so the failure presented as
    a skipped sample while the 30 Hz thread was actually frozen for minutes.

    scripts/calibrate_respeaker_doa.py passes 100 and works.
    """
    from odin_come_here.engine.audio import respeaker_doa_provider as rdp
    assert rdp._CTRL_TIMEOUT <= 1000, (
        f'_CTRL_TIMEOUT is {rdp._CTRL_TIMEOUT} ms; anything above a second '
        f'stalls the direction thread instead of failing fast'
    )
    assert rdp._CTRL_TIMEOUT > 0
