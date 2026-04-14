"""Full demo: hear 'come here' -> say 'I am coming' -> rotate toward speaker.

Uses streaming WhisperPhraseDetector with event-driven detection callback.
Critical path: speech_end -> inference -> callback -> rotate_pub
Acknowledgment playback is fire-and-forget (off critical path).
"""
import sys
import os
import time
import math
import functools
import threading
import base64

sys.path.insert(0, "/home/unitree/come-here/come_here_audio")

import rclpy
from rclpy.node import Node
from unitree_api.msg import Request
from geometry_msgs.msg import Twist
import json
import datetime
import random

from come_here_audio.whisper_phrase_detector import WhisperPhraseDetector
from come_here_audio.wake_phrase_detector import PhraseDetection
from come_here_audio.respeaker_doa_provider import ReSpeakerDOAProvider

print = functools.partial(print, flush=True)


def make_req(api_id, params=None):
    msg = Request()
    msg.header.identity.api_id = api_id
    msg.header.identity.id = int(datetime.datetime.now().timestamp() * 1000 % 2147483648) + random.randint(0, 999)
    if params is not None:
        msg.parameter = json.dumps(params) if isinstance(params, dict) else str(params)
    return msg


def play_wav_on_robot(node, pub, wav_path):
    """Send WAV to GO2 speaker via audiohub."""
    with open(wav_path, 'rb') as f:
        wav_data = f.read()
    b64 = base64.b64encode(wav_data).decode('utf-8')
    chunk_size = 16 * 1024
    chunks = [b64[i:i+chunk_size] for i in range(0, len(b64), chunk_size)]

    pub.publish(make_req(4001))
    rclpy.spin_once(node, timeout_sec=0.05)
    time.sleep(0.1)

    for i, chunk in enumerate(chunks):
        payload = {
            "current_block_index": i + 1,
            "total_block_number": len(chunks),
            "block_content": chunk,
        }
        pub.publish(make_req(4003, payload))
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.15)

    time.sleep(1.5)
    pub.publish(make_req(4002))
    rclpy.spin_once(node, timeout_sec=0.05)


def main():
    # --- ROS setup ---
    print("Init ROS...")
    rclpy.init()
    node = rclpy.create_node('come_here_demo')
    sport_pub = node.create_publisher(Request, '/api/sport/request', 10)
    audio_pub = node.create_publisher(Request, '/api/audiohub/request', 10)
    vel_pub = node.create_publisher(Twist, '/cmd_vel', 10)

    wav_path = "/home/unitree/come-here/i_am_coming.wav"

    # --- DOA setup (continuous polling) ---
    print("Starting DOA (continuous 30Hz)...")
    doa = ReSpeakerDOAProvider(frame_offset_deg=29.0)
    doa.setup()
    doa.start_continuous(poll_rate_hz=30.0)

    # --- Find ReSpeaker device ---
    # device=None selects PulseAudio "default" which returns silence.
    # Must find ReSpeaker by name. Card number changes on every USB replug.
    import sounddevice as sd
    respeaker_dev = None
    for i, dev in enumerate(sd.query_devices()):
        if 'respeaker' in dev['name'].lower() and dev['max_input_channels'] >= 6:
            respeaker_dev = i
            break
    if respeaker_dev is None:
        print("[ERROR] ReSpeaker not found! Available devices:")
        print(sd.query_devices())
        sys.exit(1)
    print(f"Found ReSpeaker at device index {respeaker_dev}")

    # --- Whisper setup (streaming) ---
    # Hardware VAD (ReSpeaker firmware) does NOT work when GO2 motors are
    # running — motor noise overwhelms it, VAD stays permanently False.
    # Instead, use hop=1000ms and let Whisper's own no_speech_prob filter
    # motor noise. This was the proven working config from 2026-04-08.
    print("Loading Whisper (base.en / cpu / int8, no VAD, hop=1000ms)...")
    detector = WhisperPhraseDetector(
        model_size="base.en",
        device="cpu",
        compute_type="int8",
        mic_device=respeaker_dev,
        mic_channels=6,
        mic_beam_channel=1,
        mic_gain=40.0,     # raw ch1 peaks ~0.03; at 25x range is ~1m, 40x extends to ~2-3m
        window_duration_s=1.5,
        hop_duration_ms=1000,  # 1s hop — limits inference rate, Whisper filters noise
        end_silence_ms=200,
        energy_threshold=0.001,
        confidence_threshold=0.30,  # motor noise degrades logprobs; Whisper hears
                                    # "come here" at 0.28-0.45 conf with motors on
        no_speech_threshold=0.75,   # default 0.5 rejects real speech when motors are
                                    # on (no_speech_prob inflated by motor noise)
        vad_check_fn=None,  # disabled — firmware VAD dead with motors running
    )

    # State shared with callback
    state = {"mode": "IDLE", "target_az": 0.0, "rotate_start": 0.0}
    latency_log = []

    def on_wake(detection: PhraseDetection, t_speech_end: float):
        """Event-driven callback: fires immediately when 'come here' detected."""
        if state["mode"] != "IDLE":
            return

        t_match = time.monotonic()
        print(f"[WAKE] Heard: '{detection.phrase}' (conf={detection.confidence:.2f})")

        # Get DOA: try latched (median of recent VAD samples), fall back to single-shot
        t_doa_start = time.monotonic()
        direction = doa.get_latched_direction(window_s=3.0)
        if direction is None:
            direction = doa.get_direction()  # single-shot fallback
        t_doa_latch = time.monotonic()

        if direction is None:
            print("[WARN] No DOA available, staying IDLE.")
            return

        target_az = direction.azimuth_rad
        deg = math.degrees(target_az)

        # Minimum angle deadzone: skip rotation for tiny DOA (noise / already facing)
        if abs(deg) < 8.0:
            print(f"[SKIP] DOA {deg:+.0f}° below 8° deadzone, already facing speaker.")
            return

        print(f"[TURN] DOA: {deg:+.0f}° (conf={direction.confidence:.2f}), rotating...")

        # Publish rotation IMMEDIATELY (critical path)
        state["mode"] = "ROTATING"
        state["target_az"] = target_az
        state["rotate_start"] = time.time()
        t_rotate_pub = time.monotonic()

        # Log latency
        latency = {
            "speech_end_to_rotate": t_rotate_pub - t_speech_end,
            "infer_to_match": t_match - t_speech_end,  # includes segmenter + inference
            "doa_latch": t_doa_latch - t_doa_start,
        }
        latency_log.append(latency)
        print(f"[LATENCY] speech_end->rotate_pub={latency['speech_end_to_rotate']:.4f}s "
              f"infer={latency['infer_to_match']:.4f}s "
              f"doa_latch={latency['doa_latch']:.4f}s")

        # Fire-and-forget acknowledgment (OFF critical path)
        def _play():
            try:
                play_wav_on_robot(node, audio_pub, wav_path)
            except Exception:
                pass
        threading.Thread(target=_play, daemon=True).start()

    detector.set_on_detection(on_wake)
    detector.setup()

    # Configure ReSpeaker AGC AFTER audio stream is open (resets on USB replug)
    try:
        import subprocess
        subprocess.run(["python3", "/home/unitree/usb_4_mic_array/tuning.py", "AGCMAXGAIN", "1000"],
                       capture_output=True, timeout=5)
        subprocess.run(["python3", "/home/unitree/usb_4_mic_array/tuning.py", "AGCONOFF", "1"],
                       capture_output=True, timeout=5)
        print("ReSpeaker AGC configured.")
    except Exception:
        pass

    print("")
    print("========================================")
    print("  COME HERE DEMO (streaming)")
    print("  Say 'come here'!")
    print("  Robot will respond and rotate to you.")
    print(f"  Calibration: DOA offset={29.0}°, "
          f"mic_gain=40.0, conf>=0.30, no_speech<=0.75")
    print(f"  Rotation: cmd_z=2.0, est 90°/s")
    print("  Running for 300 seconds (Ctrl+C to stop).")
    print("========================================")
    print("")

    start = time.time()
    try:
        while time.time() - start < 300:
            rclpy.spin_once(node, timeout_sec=0.01)

            if state["mode"] == "ROTATING":
                elapsed = time.time() - state["rotate_start"]
                target_az = state["target_az"]
                target_deg = abs(math.degrees(target_az))
                # --- Rotation calibration ---
                # Sport API Move(x,y,z): z is yaw rate. GO2 needs z>=1.5 to move.
                # cmd_z=2.0 gives smoother, more controllable rotation than 3.0.
                # Measured ~90°/s at cmd_z=2.0 on GO2 (conservative estimate).
                # If robot under-rotates: decrease deg_per_sec.
                # If robot over-rotates: increase deg_per_sec.
                cmd_z = 2.0
                deg_per_sec = 90.0
                duration = max(0.3, min(target_deg / deg_per_sec, 4.0))
                if elapsed < duration:
                    sign = 1.0 if target_az > 0 else -1.0
                    sport_pub.publish(make_req(1008, {"x": 0.0, "y": 0.0, "z": cmd_z * sign}))
                else:
                    sport_pub.publish(make_req(1003))
                    deg = math.degrees(target_az)
                    print(f"[DONE] Rotated {deg:+.0f}° in {elapsed:.1f}s "
                          f"(cmd_z={cmd_z}, est {deg_per_sec}°/s, dur={duration:.2f}s)")
                    print(f"       If under-rotated, decrease deg_per_sec. "
                          f"If over-rotated, increase deg_per_sec.")
                    print("")
                    state["mode"] = "IDLE"

            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping robot...")
        vel_pub.publish(Twist())  # stop rotation
        sport_pub.publish(make_req(1003))
        time.sleep(0.3)

        # Print latency summary
        if latency_log:
            vals = [l["speech_end_to_rotate"] for l in latency_log]
            vals.sort()
            p50 = vals[len(vals) // 2]
            p95 = vals[int(len(vals) * 0.95)]
            print(f"\n[LATENCY SUMMARY] {len(vals)} trials")
            print(f"  speech_end->rotate_pub: p50={p50:.4f}s  p95={p95:.4f}s")
            print(f"  min={min(vals):.4f}s  max={max(vals):.4f}s")

        detector.teardown()
        doa.teardown()
        node.destroy_node()
        rclpy.shutdown()
        print("Done.")


if __name__ == '__main__':
    main()
