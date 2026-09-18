"""Software DOA on synthetic plane waves: the geometry, the sign and the confidence."""

import math

import numpy as np
import pytest

from come_here_audio.ring_buffer import MultiRingBuffer
from come_here_audio.srp_doa import (
    RESPEAKER_V2_POSITIONS_M,
    SrpPhatDoa,
    synthesize_plane_wave,
    to_robot_frame,
)

RATE = 16000


def speech_like(seconds=1.0, seed=0):
    """Band-limited noise with a syllable-like envelope: voiced bursts and gaps."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    x = rng.normal(0.0, 1.0, n)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / RATE)
    spec[(f < 200) | (f > 3500)] = 0.0
    x = np.fft.irfft(spec, n=n)
    t = np.arange(n) / RATE
    envelope = 0.15 + 0.85 * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 4.0 * t)))
    x = x * envelope
    return (0.05 * x / np.max(np.abs(x))).astype(np.float32)


def angle_error_deg(a, b):
    return abs(math.degrees(math.atan2(math.sin(a - b), math.cos(a - b))))


@pytest.mark.parametrize('deg', [0, 30, 45, 90, 135, 180, -30, -90, -150, 17])
def test_recovers_the_synthetic_azimuth_within_the_grid(deg):
    doa = SrpPhatDoa(sample_rate=RATE, grid_deg=2.0)
    raw = synthesize_plane_wave(speech_like(seed=abs(deg)), math.radians(deg), RATE)
    raw += np.random.default_rng(1).normal(0.0, 0.002, raw.shape).astype(np.float32)
    est = doa.estimate(raw)
    assert est is not None
    assert angle_error_deg(est.azimuth_rad, math.radians(deg)) <= 4.0
    assert est.confidence > 0.5
    assert est.frames_used > 0


def test_an_uncorrelated_field_has_low_confidence():
    doa = SrpPhatDoa(sample_rate=RATE)
    rng = np.random.default_rng(5)
    raw = rng.normal(0.0, 0.02, (RATE, 4)).astype(np.float32)
    est = doa.estimate(raw)
    assert est is not None
    assert est.confidence < 0.3


def test_silence_and_short_clips_return_none():
    doa = SrpPhatDoa(sample_rate=RATE)
    assert doa.estimate(np.zeros((RATE, 4), dtype=np.float32)) is None
    assert doa.estimate(np.zeros((100, 4), dtype=np.float32)) is None


def test_channel_mismatch_is_a_wiring_error():
    doa = SrpPhatDoa(sample_rate=RATE)
    with pytest.raises(ValueError):
        doa.estimate(np.zeros((RATE, 3), dtype=np.float32))


def test_geometry_matches_the_odas_config():
    assert RESPEAKER_V2_POSITIONS_M == ((-0.032, 0.0), (0.0, -0.032), (0.032, 0.0), (0.0, 0.032))


def test_robot_frame_offset_and_mirror():
    assert to_robot_frame(0.0, 90.0, False) == pytest.approx(math.pi / 2)
    assert to_robot_frame(math.pi / 2, 0.0, True) == pytest.approx(-math.pi / 2)
    # Wraps: 170 deg + 30 deg offset is -160 deg, not 200.
    assert to_robot_frame(math.radians(170), 30.0, False) == pytest.approx(math.radians(-160))


def test_multi_ring_buffer_addresses_absolute_positions():
    ring = MultiRingBuffer(capacity=10, channels=2)
    ring.write(np.arange(8, dtype=np.float32).reshape(4, 2))      # positions 0..3
    ring.write(np.arange(8, 24, dtype=np.float32).reshape(8, 2))  # positions 4..11, evicts 0..1
    got = ring.read_range(2, 6)
    assert got.tolist() == [[4, 5], [6, 7], [8, 9], [10, 11]]
    evicted = ring.read_range(0, 4)
    assert len(evicted) == 2                                       # 0..1 are gone
    assert evicted.tolist() == [[4, 5], [6, 7]]
    assert len(ring.read_range(20, 30)) == 0
    with pytest.raises(ValueError):
        ring.write(np.zeros((3, 3), dtype=np.float32))


def test_estimate_from_ring_positions_matches_direct_estimate():
    """The positions the segmenter records address the same samples in the raw ring."""
    doa = SrpPhatDoa(sample_rate=RATE)
    raw = synthesize_plane_wave(speech_like(seed=3), math.radians(60), RATE)
    ring = MultiRingBuffer(capacity=6 * RATE, channels=4)
    filler = np.zeros((3 * RATE, 4), dtype=np.float32)
    ring.write(filler)
    start = ring.total_written
    for i in range(0, len(raw), 1600):
        ring.write(raw[i:i + 1600])
    end = ring.total_written
    est = doa.estimate(ring.read_range(start, end))
    assert angle_error_deg(est.azimuth_rad, math.radians(60)) <= 4.0
