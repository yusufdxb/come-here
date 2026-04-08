"""Clap-and-rotate: clap your hands, robot turns toward you.

Detects loud transient sounds (claps) via amplitude spike on the raw mic,
grabs DOA from ReSpeaker, and rotates the GO2 toward the sound source.

Separate fun/test script — does not modify any main project code.
"""

import sys, os, time, math, datetime, random, json, struct
sys.path.insert(0, "/home/unitree/come-here/come_here_audio")

import numpy as np
import sounddevice as sd
import usb.core
import usb.util
import rclpy
from unitree_api.msg import Request

# --- ReSpeaker DOA (inline, no dependency on main code) ---
VENDOR_ID = 0x2886
PRODUCT_ID = 0x0018

def read_reg(dev, reg_id, offset):
    cmd = 0x80 | offset | 0x40
    resp = dev.ctrl_transfer(
        usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
        0, cmd, reg_id, 8, 100000,
    )
    return struct.unpack(b'ii', resp.tobytes())[0]

def get_doa(dev):
    """Return (angle_deg, vad) from ReSpeaker."""
    raw = read_reg(dev, 21, 0)  # DOAANGLE
    # Apply calibration offset (29 deg) and convert to signed
    aligned = (raw + 29.0) % 360
    if aligned > 180:
        return aligned - 360, True
    return aligned, True

# --- Robot commands ---
def make_req(api_id, params=None):
    msg = Request()
    msg.header.identity.api_id = api_id
    msg.header.identity.id = int(datetime.datetime.now().timestamp() * 1000 % 2147483648) + random.randint(0, 999)
    if params is not None:
        msg.parameter = json.dumps(params)
    return msg


def main():
    print("Connecting to ReSpeaker...", flush=True)
    dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
    if dev is None:
        print("ERROR: ReSpeaker not found!")
        sys.exit(1)

    print("Init ROS...", flush=True)
    rclpy.init()
    node = rclpy.create_node('clap_rotate')
    sport_pub = node.create_publisher(Request, '/api/sport/request', 10)
    time.sleep(0.5)

    # Clap detection params
    CHUNK_SAMPLES = int(0.1 * 16000)  # 100ms chunks
    CLAP_THRESHOLD = 0.15             # amplitude threshold for clap
    COOLDOWN = 2.0                    # seconds between claps

    print("", flush=True)
    print("=" * 45, flush=True)
    print("  CLAP & ROTATE", flush=True)
    print("  Clap your hands — robot turns to you!", flush=True)
    print("  Ctrl+C to stop.", flush=True)
    print("=" * 45, flush=True)
    print("", flush=True)

    last_clap = 0
    state = "LISTENING"
    target_deg = 0.0
    rotate_start = 0.0
    clap_count = 0

    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.01)

            if state == "LISTENING":
                # Record a short chunk
                audio = sd.rec(CHUNK_SAMPLES, samplerate=16000, channels=6, dtype="float32", device="hw:0,0")
                sd.wait()
                ch = audio[:, 1].flatten()  # raw mic ch1
                peak = np.max(np.abs(ch))

                now = time.time()
                if peak > CLAP_THRESHOLD and (now - last_clap) > COOLDOWN:
                    last_clap = now
                    clap_count += 1
                    deg, _ = get_doa(dev)
                    print(f"  [CLAP #{clap_count}] peak={peak:.3f}  DOA={deg:+.0f} deg  -> rotating!", flush=True)

                    target_deg = deg
                    state = "ROTATING"
                    rotate_start = now

            elif state == "ROTATING":
                elapsed = time.time() - rotate_start
                az_rad = math.radians(abs(target_deg))
                duration = max(0.3, min(az_rad / 0.5, 5.0))

                if elapsed < duration:
                    vyaw = 0.5 if target_deg > 0 else -0.5
                    sport_pub.publish(make_req(1008, {"x": 0.0, "y": 0.0, "z": vyaw}))
                else:
                    sport_pub.publish(make_req(1003))
                    print(f"  [DONE] Rotation complete! Clap again.", flush=True)
                    print("", flush=True)
                    state = "LISTENING"

                time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping...", flush=True)
        try:
            sport_pub.publish(make_req(1003))
            time.sleep(0.3)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
        print("Done.", flush=True)


if __name__ == "__main__":
    main()
