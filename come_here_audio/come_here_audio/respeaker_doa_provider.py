"""ReSpeaker Mic Array v2.0 direction-of-arrival provider.

Uses pyusb to read DOA angle and VAD status from the XMOS XVF3000
via USB HID control transfers. The audio endpoint (ALSA) is not
touched — DOA and audio recording coexist on separate USB pipes.
"""

import math
import struct

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

    def teardown(self) -> None:
        self._dev = None
