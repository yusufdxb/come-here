"""The far-field front end: the gate that decides whether Whisper ever hears a caller.

Drives the real segmenter (WhisperPhraseDetector._process_capture) with
synthetic audio. No microphone, no Whisper model, no faster-whisper needed.
"""

import numpy as np
import pytest

from come_here_audio.whisper_phrase_detector import WhisperPhraseDetector

RATE = 16000


def make(**kwargs):
    kwargs.setdefault('mic_channels', 1)
    kwargs.setdefault('mic_beam_channel', 0)
    return WhisperPhraseDetector(**kwargs)


def noise(seconds, rms, rng):
    return rng.normal(0.0, rms, int(seconds * RATE)).astype(np.float32)


def tone(seconds, rms):
    t = np.arange(int(seconds * RATE)) / RATE
    return (np.sqrt(2.0) * rms * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def feed(detector, audio):
    """Push audio through the segmenter in 100 ms blocks, as the capture thread would."""
    segments = []
    total = detector._seg_position
    for start in range(0, len(audio), 1600):
        block = audio[start:start + 1600]
        total += len(block)
        detector._process_capture(block, total, now=total / RATE)
        try:
            segments.append(detector._segment_queue.get_nowait())
        except Exception:  # noqa: BLE001 - queue.Empty
            pass
    return segments


# -- the gate itself --

def test_a_quiet_room_lowers_the_gate_below_the_old_fixed_threshold(rng=np.random.default_rng(1)):
    d = make()
    feed(d, noise(2.0, 0.0005, rng))
    assert d._gate_floor_rms <= d.effective_rms_threshold() < 0.015


def test_a_noisy_room_raises_the_gate():
    d = make()
    feed(d, noise(2.0, 0.02, np.random.default_rng(2)))
    assert d.effective_rms_threshold() > 0.015


def test_silence_never_opens_the_gate():
    d = make()
    segments = feed(d, np.zeros(3 * RATE, dtype=np.float32))
    assert segments == []
    assert d.effective_rms_threshold() == pytest.approx(0.003)


def test_disabling_the_adaptive_gate_restores_fixed_thresholds():
    d = make(adaptive_gate=False)
    feed(d, noise(2.0, 0.0005, np.random.default_rng(3)))
    assert d.effective_rms_threshold() == 0.015
    assert d.effective_peak_threshold() == 0.02


def test_the_peak_gate_keeps_the_configured_ratio():
    d = make()
    feed(d, noise(2.0, 0.001, np.random.default_rng(4)))
    assert d.effective_peak_threshold() == pytest.approx(
        d.effective_rms_threshold() * 0.02 / 0.015)


# -- the concrete far-field case --

def test_quiet_far_field_speech_reaches_whisper_with_preroll():
    rng = np.random.default_rng(5)
    d = make()
    room = 0.0008
    speech = tone(0.6, 0.006) + noise(0.6, room, rng)  # under the old 0.015 gate
    audio = np.concatenate([noise(1.5, room, rng), speech, noise(1.2, room, rng)])
    segments = feed(d, audio)
    assert len(segments) == 1
    segment, _speech_end, _span = segments[0][:3]
    # Speech plus about 0.25 s of pre-roll plus the trailing silence that closed it.
    assert len(segment) >= int((0.6 + 0.2) * RATE)
    assert np.sqrt(np.mean(segment[:int(0.1 * RATE)] ** 2)) < 0.003  # starts before onset


def test_the_old_fixed_gate_misses_the_same_caller():
    rng = np.random.default_rng(5)
    d = make(adaptive_gate=False)
    room = 0.0008
    speech = tone(0.6, 0.006) + noise(0.6, room, rng)
    audio = np.concatenate([noise(1.5, room, rng), speech, noise(1.2, room, rng)])
    assert feed(d, audio) == []


def test_a_click_is_never_sent_to_whisper():
    rng = np.random.default_rng(6)
    d = make()
    audio = np.concatenate([noise(1.5, 0.001, rng), tone(0.05, 0.05), noise(1.5, 0.001, rng)])
    assert feed(d, audio) == []


# -- bootstrap: the room above the gate floor --

def test_ambient_above_the_gate_floor_does_not_segment_forever():
    # Measured on a ReSpeaker beam channel: ambient 0.0054 against a 0.003 floor.
    # Feeding the floor only from frames already under the gate never learns it.
    d = make()
    segments = feed(d, noise(6.0, 0.0054, np.random.default_rng(7)))
    assert segments == []
    assert d.effective_rms_threshold() > 0.0054


def test_a_room_that_gets_louder_is_re_measured():
    rng = np.random.default_rng(8)
    d = make()
    feed(d, noise(1.5, 0.0008, rng))
    feed(d, noise(8.0, 0.01, rng))  # a fan starts: one max-length utterance at most
    assert d.effective_rms_threshold() > 0.01
    assert feed(d, noise(4.0, 0.01, rng)) == []


# -- configuration --

@pytest.mark.parametrize('kwargs', [
    dict(sample_rate=44100),
    dict(mic_beam_channel=1),
    dict(gate_snr_margin=0.0),
    dict(preroll_sec=-0.1),
    dict(min_utterance_sec=5.0, max_utterance_sec=4.5),
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        make(**kwargs)


# -- software DOA rides on the same utterance --

class _FakeDoa:
    n_mics = 4

    def __init__(self):
        self.clips = []

    def estimate(self, raw):
        self.clips.append(raw)
        from come_here_audio.srp_doa import DoaEstimate
        return DoaEstimate(0.5, 0.9, 0.3, 0.8, 10, len(raw) / RATE)


def _feed_multichannel(detector, audio6):
    """Push 6-channel capture through the callback path and the segmenter."""
    total = detector._seg_position
    segments = []
    for start in range(0, len(audio6), 1600):
        block = audio6[start:start + 1600]
        detector._raw_ring.write(block[:, detector._doa_channels])
        detector._ring_buffer.write(block[:, 0].copy())
        captured, end = detector._ring_buffer.read_since(total)
        total = end
        detector._process_capture(captured, end, now=total / RATE)
        try:
            segments.append(detector._segment_queue.get_nowait())
        except Exception:  # noqa: BLE001 - queue.Empty
            pass
    return segments


def test_the_utterance_span_addresses_the_raw_capsules():
    from come_here_audio.ring_buffer import MultiRingBuffer, RingBuffer
    fake = _FakeDoa()
    d = make(mic_channels=6, mic_beam_channel=0, doa_estimator=fake, doa_channels=(1, 2, 3, 4))
    d._ring_buffer = RingBuffer(capacity=6 * RATE)
    d._raw_ring = MultiRingBuffer(6 * RATE, 4)
    d._reset_segmenter()
    rng = np.random.default_rng(9)
    quiet = noise(2.0, 0.0005, rng)
    loud = tone(0.6, 0.05)
    mono = np.concatenate([quiet, loud, noise(1.5, 0.0005, rng)])
    audio6 = np.zeros((len(mono), 6), dtype=np.float32)
    audio6[:, 0] = mono
    # The raw capsules carry a marker so the clip handed to the estimator is checkable.
    audio6[:, 1] = np.arange(len(mono), dtype=np.float32)
    segments = _feed_multichannel(d, audio6)
    assert len(segments) == 1
    segment, _t, span = segments[0][:3]
    est = d._estimate_doa(span)
    assert est is not None and d.last_doa is est
    clip = fake.clips[-1]
    assert clip.shape[1] == 4
    assert len(clip) == span[1] - span[0]
    # The marker channel proves the clip is the same instants as the utterance.
    assert clip[0, 0] == float(span[0])
    assert span[1] - span[0] >= int(0.6 * RATE)              # the whole tone is inside
    assert clip[:, 0].max() < len(quiet) + len(loud) + 1600  # and little more than it


def test_doa_channels_must_exist_in_the_capture():
    with pytest.raises(ValueError):
        make(mic_channels=1, doa_estimator=_FakeDoa(), doa_channels=(1, 2, 3, 4))
    with pytest.raises(ValueError):
        make(mic_channels=6, doa_estimator=_FakeDoa(), doa_channels=(1, 2))
