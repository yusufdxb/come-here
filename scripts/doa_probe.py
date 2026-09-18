#!/usr/bin/env python3
"""Lab stage B: where does the array think the voice is? No ROS, no robot motion.

Opens the ReSpeaker (6 channels), segments utterances with an energy gate on
the DSP beam, and prints for each one the SOFTWARE bearing from the raw
capsules (the number the demo turns on), the same bearing in the robot frame
with the offset / mirror you pass, and the firmware DOAANGLE register for
comparison. Calibration recipe (robot standing, quiet room, 1.5 m):

  1. Caller straight ahead, say "come here" 3 times.
     offset = -(median array deg)            -> launch doa_offset_deg:=<offset>
  2. Caller at the robot's LEFT (+90 deg), 3 times, rerun with --offset-deg.
     robot deg near +90: fine. Near -90: add --mirror, launch doa_mirror:=true.
  3. Caller BEHIND (180 deg) and RIGHT (-90 deg) once each: expect +/-180, -90.
  Record every line in the lab card DOA table with its confidence.

    python3 scripts/doa_probe.py                       # listen 60 s
    python3 scripts/doa_probe.py --offset-deg -35 --mirror
    python3 scripts/doa_probe.py --selftest            # synthetic, no microphone

Stop the come-here launch (and any other stack holding the microphone) first:
ALSA gives the capture device to one process.
"""

import argparse
import collections
import math
import statistics
import sys
import time

import numpy as np

RATE = 16000
FRAME = 0.05
GATE_FLOOR = 0.003
GATE_MARGIN = 2.5
SILENCE_END_S = 0.6
MIN_UTT_S = 0.25
MAX_UTT_S = 4.0


def selftest(doa) -> int:
    from come_here_audio.srp_doa import synthesize_plane_wave
    rng = np.random.default_rng(0)
    n = RATE
    x = rng.normal(0.0, 1.0, n)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / RATE)
    spec[(f < 200) | (f > 3500)] = 0
    x = 0.05 * np.fft.irfft(spec, n=n)
    worst = 0.0
    for deg in (0, 45, 90, 135, 180, -135, -90, -45):
        est = doa.estimate(synthesize_plane_wave(x, math.radians(deg), RATE))
        err = abs(math.degrees(math.atan2(math.sin(est.azimuth_rad - math.radians(deg)),
                                          math.cos(est.azimuth_rad - math.radians(deg)))))
        worst = max(worst, err)
        print(f'  synthetic {deg:+4d} deg -> {math.degrees(est.azimuth_rad):+6.1f} deg '
              f'(err {err:.1f}, conf {est.confidence:.2f})')
    print(f'selftest {"PASS" if worst <= 4.0 else "FAIL"}: worst error {worst:.1f} deg')
    return 0 if worst <= 4.0 else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seconds', type=float, default=60.0)
    parser.add_argument('--offset-deg', type=float, default=0.0)
    parser.add_argument('--mirror', action='store_true')
    parser.add_argument('--device', default='ReSpeaker')
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args(argv)

    from come_here_audio.srp_doa import (
        RESPEAKER_V2_RAW_CHANNELS, SrpPhatDoa, to_robot_frame,
    )
    doa = SrpPhatDoa(sample_rate=RATE)
    if args.selftest:
        return selftest(doa)

    import sounddevice as sd
    from come_here_audio.mic_select import resolve
    index, name, channels, far_field = resolve(args.device, True)
    if index is None or not far_field:
        print(f'need the 6-channel ReSpeaker, got {name} ({channels}ch)')
        return 1
    print(f'mic {index}:{name} ({channels}ch)  offset {args.offset_deg:+.1f} deg  '
          f'mirror {args.mirror}')

    firmware = None
    try:
        from come_here_audio import respeaker_tune
        firmware = respeaker_tune.find_device()
    except Exception as exc:  # noqa: BLE001
        print(f'firmware DOAANGLE unavailable: {exc}')

    raw_ch = list(RESPEAKER_V2_RAW_CHANNELS)
    frame_n = int(FRAME * RATE)
    pending = np.zeros((0, 6), dtype=np.float32)
    levels = collections.deque(maxlen=int(3.0 / FRAME))
    utterance = []
    last_speech_frames = 0
    count = 0
    array_history = []

    def report(block):
        nonlocal count
        count += 1
        raw = block[:, raw_ch]
        est = doa.estimate(raw)
        fw = None
        if firmware is not None:
            try:
                fw = int(respeaker_tune.read_parameter(firmware, 'DOAANGLE'))
            except Exception:  # noqa: BLE001
                fw = None
        if est is None:
            print(f'#{count:2d} {len(block) / RATE:4.2f}s  no estimate'
                  + (f'  firmware {fw}' if fw is not None else ''))
            return
        robot = to_robot_frame(est.azimuth_rad, args.offset_deg, args.mirror)
        array_history.append(math.degrees(est.azimuth_rad))
        print(f'#{count:2d} {len(block) / RATE:4.2f}s  array {math.degrees(est.azimuth_rad):+6.1f} deg'
              f'  robot {math.degrees(robot):+6.1f} deg  conf {est.confidence:.2f}'
              f'  peak {est.peak:.3f} contrast {est.contrast:.2f}'
              + (f'  firmware {fw:3d}' if fw is not None else ''))

    def callback(indata, frames, t, status):
        nonlocal pending
        pending = np.concatenate((pending, indata.astype(np.float32)))

    stream = sd.InputStream(samplerate=RATE, channels=6, dtype='float32', device=index,
                            blocksize=int(0.1 * RATE), callback=callback)
    stream.start()
    print(f'listening {args.seconds:.0f} s: say "come here" (Ctrl+C to stop)')
    end = time.monotonic() + args.seconds
    try:
        while time.monotonic() < end:
            time.sleep(0.05)
            block, pending = pending, np.zeros((0, 6), dtype=np.float32)
            for i in range(0, len(block) - frame_n + 1, frame_n):
                frame = block[i:i + frame_n]
                rms = float(np.sqrt(np.mean(frame[:, 0] ** 2)))
                levels.append(rms)
                floor = float(np.percentile(levels, 25)) if len(levels) >= 10 else rms
                gate = max(GATE_FLOOR, floor * GATE_MARGIN)
                voiced = rms >= gate
                if not utterance:
                    if voiced:
                        utterance.append(frame)
                        last_speech_frames = 0
                    continue
                utterance.append(frame)
                last_speech_frames = 0 if voiced else last_speech_frames + 1
                length_s = len(utterance) * FRAME
                if last_speech_frames * FRAME >= SILENCE_END_S or length_s >= MAX_UTT_S:
                    spoken_s = length_s - last_speech_frames * FRAME
                    if spoken_s >= MIN_UTT_S:
                        report(np.concatenate(utterance))
                    utterance.clear()
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
    if array_history:
        med = statistics.median(array_history)
        print(f'{len(array_history)} estimates, median array bearing {med:+.1f} deg')
        print(f'if the caller was straight ahead: doa_offset_deg:={-med:.1f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
