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

    def get_latched_direction(
        self,
        window_s: float = 1.0,
        min_samples: int = 3,
        agreement_thresh_rad: float = 0.35,  # ~20 degrees
    ) -> Optional[DirectionEstimate]:
        """Return the median DOA from VAD-active samples, with outlier rejection.

        Uses iterative filtering: compute circular median, discard samples
        farther than agreement_thresh_rad, recompute. Confidence reflects
        how tightly the remaining samples agree.

        Returns None if no VAD-active samples exist or if fewer than
        min_samples agree on a direction.

        Args:
            window_s: Time window in seconds to consider.
            min_samples: Minimum agreeing samples required to return a result.
            agreement_thresh_rad: Max angular distance from median to keep a sample.
        """
        now = time.monotonic()
        cutoff = now - window_s

        # Collect VAD-active samples within the window
        active_az = [
            az for t, az, vad in self._samples
            if vad and t >= cutoff
        ]

        if len(active_az) < min_samples:
            return None

        # Circular median: convert to unit vectors, average, then atan2
        def _circular_median(angles):
            xs = [math.cos(a) for a in angles]
            ys = [math.sin(a) for a in angles]
            return math.atan2(np.median(ys), np.median(xs))

        def _angular_distance(a, b):
            """Shortest angular distance between two angles in radians."""
            d = abs(a - b) % (2 * math.pi)
            return min(d, 2 * math.pi - d)

        # Iterative outlier rejection (2 passes)
        remaining = list(active_az)
        for _ in range(2):
            if len(remaining) < min_samples:
                return None
            center = _circular_median(remaining)
            remaining = [
                a for a in remaining
                if _angular_distance(a, center) <= agreement_thresh_rad
            ]

        if len(remaining) < min_samples:
            return None

        final_az = _circular_median(remaining)

        # Confidence: fraction of original samples that survived filtering
        survival_ratio = len(remaining) / len(active_az)
        # Spread: mean angular distance from center (lower = more confident)
        spread = np.mean([_angular_distance(a, final_az) for a in remaining])
        # Combine: high survival + low spread = high confidence
        confidence = min(0.95, survival_ratio * (1.0 - spread / agreement_thresh_rad))
        confidence = max(0.3, confidence)

        return DirectionEstimate(azimuth_rad=final_az, confidence=confidence)

    def teardown(self) -> None:
        self._polling = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None
        self._dev = None
