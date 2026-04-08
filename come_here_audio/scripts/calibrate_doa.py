"""DOA calibration: stand in front of the robot and speak to find frame_offset_deg.

Usage:
    python3 calibrate_doa.py

Stand directly in front of the robot (facing its face/head) and speak or clap
repeatedly. The script collects raw DOA readings when VAD is active, and also
shows the raw angle continuously so you can see the mic responding.

Press Ctrl+C when you have enough samples (aim for 10+).
"""

import struct
import sys
import time
import math

import usb.core
import usb.util

VENDOR_ID = 0x2886
PRODUCT_ID = 0x0018
CTRL_TIMEOUT = 100000

REG_DOAANGLE = (21, 0)
REG_VOICEACTIVITY = (19, 32)


def read_register(dev, reg_id, offset, is_int=True):
    cmd = 0x80 | offset
    if is_int:
        cmd |= 0x40
    response = dev.ctrl_transfer(
        usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
        0, cmd, reg_id, 8, CTRL_TIMEOUT,
    )
    result = struct.unpack(b'ii', response.tobytes())
    return result[0] if is_int else result[0] * (2.0 ** result[1])


def circular_mean(angles_deg):
    """Compute mean of angles on a circle (handles wraparound)."""
    rads = [math.radians(a) for a in angles_deg]
    x = sum(math.cos(r) for r in rads) / len(rads)
    y = sum(math.sin(r) for r in rads) / len(rads)
    mean_rad = math.atan2(y, x)
    return math.degrees(mean_rad) % 360


def main():
    dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
    if dev is None:
        print("ERROR: ReSpeaker not found!")
        sys.exit(1)
    print("ReSpeaker found.")
    print()
    print("=" * 55)
    print("  DOA CALIBRATION")
    print("  Stand DIRECTLY IN FRONT of the robot.")
    print("  Speak or clap repeatedly.")
    print("  VAD samples are collected; raw DOA shown always.")
    print("  Press Ctrl+C when done (aim for 10+ VAD samples).")
    print("=" * 55)
    print()

    samples = []
    tick = 0
    try:
        while True:
            vad = read_register(dev, *REG_VOICEACTIVITY)
            raw_deg = read_register(dev, *REG_DOAANGLE)

            if vad:
                samples.append(raw_deg)
                print(f"  *** [{len(samples):3d}] VAD=1  raw_DOA = {raw_deg:3d} deg  <<<")
            else:
                # Show raw DOA every 5th tick (~0.5s) to reduce noise
                if tick % 5 == 0:
                    print(f"       VAD=0  raw_DOA = {raw_deg:3d} deg")

            tick += 1
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    print()
    if len(samples) < 3:
        print("Not enough VAD samples (got %d, need at least 3). Try again." % len(samples))
        sys.exit(1)

    mean_raw = circular_mean(samples)
    # frame_offset_deg: what to ADD to raw so that "front" maps to 0
    offset = (-mean_raw) % 360
    if offset > 180:
        offset -= 360

    print(f"Collected {len(samples)} VAD samples.")
    print(f"Raw angles: min={min(samples)}, max={max(samples)}")
    print(f"Circular mean of raw DOA when facing front: {mean_raw:.1f} deg")
    print()
    print(f"  >>> frame_offset_deg = {offset:.1f}")
    print()
    print("Use this value in ReSpeakerDOAProvider and hear_and_rotate_demo.py")


if __name__ == "__main__":
    main()
