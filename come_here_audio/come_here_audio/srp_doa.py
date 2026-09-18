"""Software direction of arrival from the ReSpeaker Mic Array v2.0 raw capsules.

Why software: the array's firmware DOAANGLE register was dead for a week in
April 2026 and was seen stuck again in September. This estimator reads the
four raw capsules that the 6-channel firmware streams on capture channels
1..4 and computes the direction itself, from the same samples Whisper heard,
so the bearing belongs to the utterance that woke the robot and to nothing
else. There is no USB control transfer, no firmware VAD and no timing race.

Method: SRP-PHAT (steered response power with phase transform). For every
microphone pair the cross-spectrum is normalised to unit magnitude (PHAT),
accumulated over the voiced frames of the utterance, and steered over a
grid of candidate azimuths; the azimuth whose steering phases line up best
across all six pairs wins. PHAT makes the estimate insensitive to the
loudness and spectrum of the talker, which is what a wake phrase needs.

Geometry (metres, array frame) for capture channels 1..4, taken from the
ODAS ``respeaker_usb_4_mic_array.cfg``: (-0.032, 0), (0, -0.032),
(0.032, 0), (0, 0.032). Azimuth 0 is the +x axis of that frame; the mount
offset and mirror (array facing down) are applied by ``to_robot_frame``.

Everything here is numpy only, so it is unit-tested with synthetic delays.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

SPEED_OF_SOUND_M_S = 343.0
RESPEAKER_V2_POSITIONS_M = ((-0.032, 0.0), (0.0, -0.032), (0.032, 0.0), (0.0, 0.032))
RESPEAKER_V2_RAW_CHANNELS = (1, 2, 3, 4)


def wrap_pi(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def to_robot_frame(azimuth_array_rad: float, frame_offset_deg: float, mirror: bool) -> float:
    """Array-frame azimuth -> ROS convention (0 forward, positive left).

    ``mirror`` flips the turning sense for an array mounted capsules-down.
    ``frame_offset_deg`` is added afterwards: the azimuth the estimator reports
    for a caller straight ahead, negated. Both are calibrated with
    ``scripts/doa_probe.py``.
    """
    a = -azimuth_array_rad if mirror else azimuth_array_rad
    return wrap_pi(a + math.radians(frame_offset_deg))


@dataclass(frozen=True)
class DoaEstimate:
    azimuth_rad: float      # array frame, before offset / mirror
    confidence: float       # 0..1, from peak height and map contrast
    peak: float             # normalised SRP peak, 1.0 = perfect coherence
    contrast: float         # (peak - mean) / peak over the azimuth grid
    frames_used: int
    seconds: float

    def as_dict(self) -> dict:
        return {
            'doa_array_deg': round(math.degrees(self.azimuth_rad), 1),
            'doa_confidence': round(self.confidence, 3),
            'doa_peak': round(self.peak, 4),
            'doa_contrast': round(self.contrast, 3),
            'doa_frames': self.frames_used,
            'doa_seconds': round(self.seconds, 2),
        }


class SrpPhatDoa:
    """SRP-PHAT azimuth estimator for a planar array.

    Args:
        sample_rate: capture rate; the ReSpeaker streams 16 kHz.
        positions: (x, y) metres per raw channel, in capture-channel order.
        grid_deg: azimuth grid step. 2 degrees is plenty: the visual ALIGN
            takes over once the caller is inside the camera view.
        frame_size, hop: STFT framing in samples.
        band_hz: bins outside this band are ignored. The upper edge stays
            under the spatial-aliasing limit of the 64 mm diagonal pairs.
        peak_full_scale: normalised peak height that counts as full
            confidence. Real rooms sit well below 1.0 (reverberation and
            the robot's own noise); tune from doa_probe.py readings.
        voiced_fraction: only frames at or above this fraction of the loudest
            frame's RMS enter the estimate, so the trailing silence of an
            utterance does not dilute it.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        positions: Sequence[Sequence[float]] = RESPEAKER_V2_POSITIONS_M,
        grid_deg: float = 2.0,
        frame_size: int = 512,
        hop: int = 256,
        band_hz: Sequence[float] = (300.0, 3000.0),
        speed_of_sound: float = SPEED_OF_SOUND_M_S,
        peak_full_scale: float = 0.25,
        voiced_fraction: float = 0.3,
        min_seconds: float = 0.15,
    ):
        if len(positions) < 2:
            raise ValueError('need at least two microphones')
        if not 0 < grid_deg <= 30:
            raise ValueError('grid_deg must be in (0, 30]')
        if hop <= 0 or frame_size < hop:
            raise ValueError('frame_size must be >= hop > 0')
        self._fs = int(sample_rate)
        self._pos = np.asarray(positions, dtype=np.float64)
        self._n_mics = len(self._pos)
        self._frame = int(frame_size)
        self._hop = int(hop)
        self._peak_full_scale = float(peak_full_scale)
        self._voiced_fraction = float(voiced_fraction)
        self._min_samples = int(min_seconds * self._fs)

        self._angles = np.deg2rad(np.arange(0.0, 360.0, float(grid_deg)))
        unit = np.stack([np.cos(self._angles), np.sin(self._angles)], axis=1)  # (A, 2)
        self._pairs = [(i, j) for i in range(self._n_mics) for j in range(i + 1, self._n_mics)]
        # Plane wave from azimuth theta: mic i receives at -(p_i . u) / c
        # relative to the origin, so the pair delay tau_i - tau_j is
        # -((p_i - p_j) . u) / c and the compensation phase is +2 pi f tau.
        diffs = np.array([self._pos[i] - self._pos[j] for i, j in self._pairs])  # (P, 2)
        tau = -(diffs @ unit.T) / float(speed_of_sound)                           # (P, A)

        freqs = np.fft.rfftfreq(self._frame, 1.0 / self._fs)
        self._band = np.where((freqs >= band_hz[0]) & (freqs <= band_hz[1]))[0]
        if len(self._band) < 4:
            raise ValueError('band_hz selects too few bins')
        f = freqs[self._band]                                                     # (B,)
        self._steer = np.exp(2j * np.pi * tau[:, :, None] * f[None, None, :]).astype(np.complex64)
        self._window = np.hanning(self._frame).astype(np.float32)

    @property
    def n_mics(self) -> int:
        return self._n_mics

    @property
    def grid_deg(self) -> float:
        return float(np.rad2deg(self._angles[1] - self._angles[0]))

    def estimate(self, raw: np.ndarray) -> Optional[DoaEstimate]:
        """Azimuth of the dominant source in ``raw`` (samples x mics), or None.

        Returns None when the clip is too short or silent. Raises ValueError
        on a channel-count mismatch, which is a wiring bug, not a quiet room.
        """
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != self._n_mics:
            raise ValueError(f'expected (n, {self._n_mics}) samples, got {raw.shape}')
        n = raw.shape[0]
        if n < max(self._frame, self._min_samples):
            return None

        n_frames = 1 + (n - self._frame) // self._hop
        idx = np.arange(self._frame)[None, :] + self._hop * np.arange(n_frames)[:, None]
        frames = raw[idx] * self._window[None, :, None]          # (F, N, M)
        rms = np.sqrt(np.mean(frames ** 2, axis=(1, 2)))          # (F,)
        loudest = float(rms.max())
        if loudest <= 0.0:
            return None
        keep = rms >= self._voiced_fraction * loudest
        frames = frames[keep]
        spec = np.fft.rfft(frames, axis=1)[:, self._band, :]     # (F, B, M)

        acc = np.zeros((len(self._pairs), len(self._band)), dtype=np.complex64)
        for p, (i, j) in enumerate(self._pairs):
            cross = spec[:, :, i] * np.conj(spec[:, :, j])       # (F, B)
            mag = np.abs(cross)
            phat = np.where(mag > 1e-12, cross / np.maximum(mag, 1e-12), 0.0)
            acc[p] = phat.sum(axis=0)
        acc /= max(1, frames.shape[0])

        # Steered response, normalised so perfect coherence gives 1.0.
        srp = np.real(np.einsum('pb,pab->a', acc, self._steer)) / (len(self._pairs) * len(self._band))
        k = int(np.argmax(srp))
        peak = float(srp[k])
        if peak <= 0.0:
            return None
        # Parabolic refinement between grid points (the grid is circular).
        left, right = srp[k - 1], srp[(k + 1) % len(srp)]
        denom = left - 2.0 * peak + right
        shift = 0.0 if abs(denom) < 1e-12 else 0.5 * (left - right) / denom
        shift = max(-1.0, min(1.0, shift))
        step = self._angles[1] - self._angles[0]
        azimuth = wrap_pi(float(self._angles[k] + shift * step))

        contrast = max(0.0, (peak - float(srp.mean())) / peak)
        confidence = max(0.0, min(1.0, contrast * min(1.0, peak / self._peak_full_scale)))
        return DoaEstimate(
            azimuth_rad=azimuth,
            confidence=confidence,
            peak=peak,
            contrast=contrast,
            frames_used=int(frames.shape[0]),
            seconds=n / self._fs,
        )


def synthesize_plane_wave(source: np.ndarray, azimuth_rad: float, sample_rate: int = 16000,
                          positions: Sequence[Sequence[float]] = RESPEAKER_V2_POSITIONS_M,
                          speed_of_sound: float = SPEED_OF_SOUND_M_S) -> np.ndarray:
    """Delay ``source`` onto each microphone for a far-field talker at ``azimuth_rad``.

    Fractional delays are applied in the frequency domain. Used by the tests
    and by doa_probe.py --selftest; it is the inverse model of ``estimate``.
    """
    source = np.asarray(source, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    u = np.array([math.cos(azimuth_rad), math.sin(azimuth_rad)])
    spectrum = np.fft.rfft(source)
    freqs = np.fft.rfftfreq(len(source), 1.0 / sample_rate)
    out = np.empty((len(source), len(pos)), dtype=np.float32)
    for m, p in enumerate(pos):
        tau = -float(p @ u) / speed_of_sound
        out[:, m] = np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * tau), n=len(source))
    return out
