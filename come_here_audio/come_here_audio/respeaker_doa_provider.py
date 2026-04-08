"""ReSpeaker Mic Array v2.0 direction-of-arrival provider.

Uses pyusb to read DOA angle and VAD status from the XMOS XVF3000
via USB HID control transfers. The audio endpoint (ALSA) is not
touched — DOA and audio recording coexist on separate USB pipes.

Supports both single-shot polling (get_direction) and continuous
background polling (start_continuous / get_latched_direction) for
low-latency DOA capture.
"""

import collections
import math
import struct
import threading
import time
from typing import Optional

import numpy as np
import usb.core
import usb.util

from come_here_audio.audio_direction_provider import AudioDirectionProvider, DirectionEstimate

# ReSpeaker Mic Array v2.0 USB IDs
_VENDOR_ID = 0x2886
_PRODUCT_ID = 0x0018

# XMOS register addresses (from tuning.py PARAMETERS table)
_REG_DOAANGLE = (21, 0)       # (id=21, offset=0), int, 0-359 degrees
_REG_VOICEACTIVITY = (19, 32)  # (id=19, offset=32), int, 0 or 1

_CTRL_TIMEOUT = 100000  # microseconds

# AGC register addresses (from tuning.py PARAMETERS table)
_REG_AGCONOFF = (19, 0)       # AGC enable: 0=off, 1=on
_REG_AGCMAXGAIN = (19, 4)     # AGC max gain: 0-1000


def _read_register(dev, reg_id: int, offset: int, is_int: bool = True):
    """Read a single register from the XMOS via USB control transfer."""
    cmd = 0x80 | offset
    if is_int:
        cmd |= 0x40
    response = dev.ctrl_transfer(
        usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
        0, cmd, reg_id, 8, _CTRL_TIMEOUT,
    )
    result = struct.unpack(b'ii', response.tobytes())
    return result[0] if is_int else result[0] * (2.0 ** result[1])


def _write_register(dev, reg_id: int, offset: int, value: int) -> None:
    """Write an integer register to the XMOS via USB control transfer."""
    cmd = offset | 0x40
    data = struct.pack(b'ii', value, 0)
    dev.ctrl_transfer(
        usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
        0, cmd, reg_id, data, _CTRL_TIMEOUT,
    )


class ReSpeakerDOAProvider(AudioDirectionProvider):
    """Direction-of-arrival from the ReSpeaker Mic Array v2.0.

    Reads DOAANGLE (0-359 degrees, clockwise from mic 0) and
    VOICEACTIVITY (0/1) at each poll. Returns None when VAD is
    inactive. Converts degrees to radians in ROS convention
    (0 = forward, positive = left / counter-clockwise) with a
    configurable mount offset.

    Args:
        frame_offset_deg: Degrees to add to raw DOA to align mic 0
            with the robot's forward axis. Positive = counter-clockwise.
    """

    def __init__(self, frame_offset_deg: float = 29.0):
        self._frame_offset_deg = frame_offset_deg
        self._dev = None
        # Continuous polling state
        self._polling = False
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_interval = 1.0 / 30.0
        self._samples: collections.deque = collections.deque(maxlen=300)  # ~10s at 30Hz

    def setup(self) -> None:
        self._dev = usb.core.find(idVendor=_VENDOR_ID, idProduct=_PRODUCT_ID)
        if self._dev is None:
            raise RuntimeError(
                f"ReSpeaker Mic Array not found (VID={_VENDOR_ID:#06x}, "
                f"PID={_PRODUCT_ID:#06x}). Check USB connection and udev rules."
            )

    def get_direction(self) -> DirectionEstimate | None:
        if self._dev is None:
            return None

        vad = _read_register(self._dev, *_REG_VOICEACTIVITY, is_int=True)
        if not vad:
            return None

        raw_deg = _read_register(self._dev, *_REG_DOAANGLE, is_int=True)

        # Apply mount offset and convert to ROS convention:
        # ReSpeaker: 0-359 clockwise from mic 0
        # ROS: 0 = forward, positive = left (counter-clockwise)
        aligned_deg = (raw_deg + self._frame_offset_deg) % 360
        if aligned_deg > 180:
            azimuth_rad = -math.radians(360 - aligned_deg)
        else:
            azimuth_rad = math.radians(aligned_deg)

        return DirectionEstimate(
            azimuth_rad=azimuth_rad,
            confidence=0.9,  # fixed for now; VAD is binary
        )

    def start_continuous(self, poll_rate_hz: float = 30.0) -> None:
        """Start continuous background DOA polling.

        Stores timestamped (time, azimuth_rad, vad_active) tuples in a
        deque for low-latency direction latching.

        Args:
            poll_rate_hz: Polling rate in Hz. 30Hz is the max reliable
                rate for the ReSpeaker USB HID interface.
        """
        if self._polling:
            return
        self._poll_interval = 1.0 / poll_rate_hz
        self._polling = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="doa_poll"
        )
        self._poll_thread.start()

    def _poll_loop(self) -> None:
        """Background thread: poll DOA + VAD at the configured rate."""
        while self._polling and self._dev is not None:
            try:
                vad = _read_register(self._dev, *_REG_VOICEACTIVITY, is_int=True)
                raw_deg = _read_register(self._dev, *_REG_DOAANGLE, is_int=True)

                aligned_deg = (raw_deg + self._frame_offset_deg) % 360
                if aligned_deg > 180:
                    azimuth_rad = -math.radians(360 - aligned_deg)
                else:
                    azimuth_rad = math.radians(aligned_deg)

                self._samples.append((time.monotonic(), azimuth_rad, bool(vad)))
            except Exception:
                pass  # USB glitch — skip this sample

            time.sleep(self._poll_interval)

    def get_latched_direction(self, window_s: float = 1.0) -> Optional[DirectionEstimate]:
        """Return the median DOA from recent samples in the last window_s seconds.

        Prefers VAD-active samples but falls back to all samples if no
        VAD-active ones exist (the DOA register retains the last direction
        even after voice activity ends).

        Returns None only if no samples exist at all.
        """
        now = time.monotonic()
        cutoff = now - window_s

        # Try VAD-active samples first
        active = [(t, az) for t, az, vad in self._samples if vad and t >= cutoff]

        if len(active) >= 3:
            azimuths = [az for _, az in active]
            median_az = float(np.median(azimuths))
            confidence = min(0.95, 0.6 + 0.05 * len(active))
            return DirectionEstimate(azimuth_rad=median_az, confidence=confidence)

        # Fall back: use ALL samples in window (DOA register holds last direction)
        all_samples = [(t, az) for t, az, _vad in self._samples if t >= cutoff]

        if not all_samples:
            return None

        # Use the most recent sample
        _, az = all_samples[-1]
        confidence = 0.4 if not active else 0.6
        return DirectionEstimate(azimuth_rad=az, confidence=confidence)

    def teardown(self) -> None:
        self._polling = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None
        self._dev = None
