"""Full demo: hear 'come here' -> say 'I am coming' -> rotate toward speaker."""
import sys, os, time, math, functools, threading, queue, base64
sys.path.insert(0, "/home/unitree/come-here/come_here_audio")

import rclpy
from rclpy.node import Node
from unitree_api.msg import Request
import json, datetime, random
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
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


# Whisper background thread
detection_q = queue.Queue()

def whisper_thread(model):
    while True:
        try:
            audio = sd.rec(int(3 * 16000), samplerate=16000, channels=6, dtype="float32", device="hw:0,0")
            sd.wait()
            ch = audio[:, 1].flatten()
            ch = np.clip(ch * 4.0, -1.0, 1.0)
            if np.max(np.abs(ch)) < 0.02:
                continue
            segs, info = model.transcribe(ch, beam_size=1, language="en")
            for s in segs:
                if s.no_speech_prob > 0.5:
                    continue
                if "come here" in s.text.strip().lower():
                    detection_q.put(s.text.strip())
        except Exception:
            time.sleep(1)


def main():
    print("Loading Whisper...")
    model_path = "/home/unitree/come-here/models/faster-whisper-base.en/models--Systran--faster-whisper-base.en/snapshots/3d3d5dee26484f91867d81cb899cfcf72b96be6c"
    model = WhisperModel(model_path, device="cpu", compute_type="int8")

    print("Starting DOA...")
    doa = ReSpeakerDOAProvider(frame_offset_deg=0.0)
    doa.setup()

    print("Starting Whisper listener...")
    t = threading.Thread(target=whisper_thread, args=(model,), daemon=True)
    t.start()

    print("Init ROS...")
    rclpy.init()
    node = rclpy.create_node('come_here_demo')
    sport_pub = node.create_publisher(Request, '/api/sport/request', 10)
    audio_pub = node.create_publisher(Request, '/api/audiohub/request', 10)

    wav_path = "/home/unitree/come-here/i_am_coming.wav"

    state = "IDLE"
    target_az = 0.0
    rotate_start = 0.0
    last_doa = None

    print("")
    print("========================================")
    print("  COME HERE DEMO - Say 'come here'!")
    print("  Robot will respond and rotate to you.")
    print("  Running for 120 seconds.")
    print("========================================")
    print("")

    start = time.time()
    try:
        while time.time() - start < 120:
            rclpy.spin_once(node, timeout_sec=0.01)

            d = doa.get_direction()
            if d:
                last_doa = d

            if state == "IDLE":
                try:
                    txt = detection_q.get_nowait()
                    print("[WAKE] Heard: '%s'" % txt)

                    # Say "I am coming"
                    print("[SPEAK] Playing 'I am coming'...")
                    play_wav_on_robot(node, audio_pub, wav_path)
                    print("[SPEAK] Done.")

                    # Rotate toward speaker
                    if last_doa:
                        target_az = last_doa.azimuth_rad
                        deg = math.degrees(target_az)
                        print("[TURN] DOA: %+.0f deg, rotating..." % deg)
                        state = "ROTATING"
                        rotate_start = time.time()
                    else:
                        print("[WARN] No DOA available, back to listening.")
                except queue.Empty:
                    pass

            elif state == "ROTATING":
                elapsed = time.time() - rotate_start
                duration = max(0.5, min(abs(target_az) / 0.5, 5.0))
                if elapsed < duration:
                    vyaw = 0.5 if target_az > 0 else -0.5
                    sport_pub.publish(make_req(1008, {"x": 0.0, "y": 0.0, "z": vyaw}))
                else:
                    sport_pub.publish(make_req(1003))
                    print("[DONE] Rotation complete! Say 'come here' again.")
                    print("")
                    state = "IDLE"
                    last_doa = None

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping robot...")
        sport_pub.publish(make_req(1003))
        time.sleep(0.3)
        doa.teardown()
        node.destroy_node()
        rclpy.shutdown()
        print("Done.")


if __name__ == '__main__':
    main()
