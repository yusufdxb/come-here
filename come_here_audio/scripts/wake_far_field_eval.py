#!/usr/bin/env python3
"""How far away can come-here still hear "come here", and what rejects it when it cannot.

Degrades a clean clip the way distance does, streams it in real time through
the REAL WhisperPhraseDetector, and reports which stage rejected each clip:

    gate     the energy gate never opened, so Whisper never saw it
    whisper  a segment reached Whisper but nothing matched (or it was rejected
             on confidence / no_speech)
    HEARD    "come here" fired

Negative phrases (--negatives) are scored the other way: any detection is a
false trigger.

Distance costs two things at once: --levels is attenuation in dB (about 6 dB
per doubling of distance in free field) and --snrs is the noise floor relative
to the speech. --reverb-s adds a decaying tail.

    python3 come_here_audio/scripts/wake_far_field_eval.py               # adaptive gate
    python3 come_here_audio/scripts/wake_far_field_eval.py --fixed-gate  # previous gate
    python3 come_here_audio/scripts/wake_far_field_eval.py --wav-dir rec/ # real 16 kHz clips

THIS IS A PROXY, NOT THE ROBOT. Synthetic attenuation and stationary noise are
not a real room and not GO2 motor noise. It shows which STAGE gives up first
and whether a change moves that boundary; only the robot gives metres.
"""

import argparse
import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from come_here_audio.whisper_phrase_detector import WhisperPhraseDetector  # noqa: E402

RATE = 16000


def synthesise(phrase: str, out_dir: Path, voice: str) -> Path:
    raw = out_dir / f'{phrase.replace(" ", "_")}_{voice}_raw.wav'
    out = out_dir / f'{phrase.replace(" ", "_")}_{voice}.wav'
    subprocess.run(['espeak-ng', '-v', voice, '-w', str(raw), '-s', '140', phrase],
                   check=True, capture_output=True)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(raw),
                    '-ar', str(RATE), '-ac', '1', '-c:a', 'pcm_s16le', str(out)], check=True)
    return out


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), 'rb') as handle:
        if handle.getframerate() != RATE:
            raise SystemExit(f'{path} must be {RATE} Hz')
        frames = handle.readframes(handle.getnframes())
        channels = handle.getnchannels()
    audio = np.frombuffer(frames, dtype='<i2').astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels)[:, 0]
    return audio


def add_reverb(audio, rt60_s, rng):
    length = int(rt60_s * RATE)
    if length < 2:
        return audio
    times = np.arange(length) / RATE
    impulse = rng.normal(0, 1, length) * np.exp(-6.9 * times / rt60_s)
    impulse[0] += 1.0
    impulse /= np.sqrt(np.sum(impulse ** 2))
    return np.convolve(audio, impulse)[:len(audio) + length]


def degrade(clean, level_db, snr_db, reverb_s, rng):
    signal = clean * (10 ** (level_db / 20.0))
    if reverb_s > 0:
        signal = add_reverb(signal, reverb_s, rng)
    speech_rms = float(np.sqrt(np.mean(signal ** 2))) or 1e-9
    noise_rms = speech_rms / (10 ** (snr_db / 20.0))
    white = rng.normal(0, 1, len(signal) + 3 * RATE).astype(np.float32)
    pink = np.copy(white)
    for index in range(1, len(pink)):
        pink[index] = 0.98 * pink[index - 1] + 0.02 * white[index]
    pink /= (float(np.sqrt(np.mean(pink ** 2))) or 1e-9)
    pink *= noise_rms
    lead = np.zeros(int(1.5 * RATE), dtype=np.float32)
    tail = np.zeros(int(1.5 * RATE), dtype=np.float32)
    padded = np.concatenate([lead, signal, tail])
    return (padded + pink[:len(padded)]).astype(np.float32)


def drain(detector):
    deadline = time.monotonic() + 15.0
    quiet_since = time.monotonic()
    while time.monotonic() < deadline:
        drained = False
        for q in (detector._segment_queue, detector._detections):
            while True:
                try:
                    q.get_nowait()
                    drained = True
                except Exception:  # noqa: BLE001 - queue.Empty
                    break
        if drained:
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since > 1.5:
            return
        time.sleep(0.1)


def run_clip(detector, audio, log):
    drain(detector)
    clip_start = time.monotonic()
    detections = []

    def on_detection(detection, speech_end_s=0.0):
        if speech_end_s and speech_end_s < clip_start - 0.2:
            return  # belongs to the previous clip
        detections.append((detection, time.monotonic() - speech_end_s))

    detector.set_on_detection(on_detection)
    block = int(0.1 * RATE)
    for start in range(0, len(audio), block):
        chunk = audio[start:start + block].reshape(-1, 1)
        detector._audio_callback(chunk, len(chunk), None, None)
        time.sleep(0.1)
    silence = np.zeros((block, 1), dtype=np.float32)
    for _ in range(12):
        detector._audio_callback(silence, len(silence), None, None)
        time.sleep(0.1)
    deadline = time.monotonic() + 6.0
    while not detections and time.monotonic() < deadline:
        time.sleep(0.05)
    text = log()
    if detections:
        return 'HEARD', detections[0]
    if '[SEG] utterance closed' in text and 'rejected peak' not in text:
        return 'whisper', None
    return 'gate', None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--phrase', default='come here')
    parser.add_argument('--negatives', default='i am here,i am coming,come on',
                        help='comma-separated phrases that must NOT trigger')
    parser.add_argument('--voices', default='en-us,en')
    parser.add_argument('--wav-dir', type=Path, default=None,
                        help='real 16 kHz clips named come_here.wav etc.')
    parser.add_argument('--levels', default='0,-12,-20,-26')
    parser.add_argument('--snrs', default='30,15,8')
    parser.add_argument('--reverb-s', type=float, default=0.4)
    parser.add_argument('--fixed-gate', action='store_true', help='the previous fixed 0.015 gate')
    parser.add_argument('--confidence', type=float, default=0.4)
    parser.add_argument('--no-speech', type=float, default=0.5)
    parser.add_argument('--model', default='base.en')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args(argv)

    work = Path(tempfile.mkdtemp(prefix='come_here_far_field_'))
    voices = [v for v in args.voices.split(',') if v]
    negatives = [n.strip() for n in args.negatives.split(',') if n.strip()]

    def clips_for(phrase):
        if args.wav_dir is not None:
            path = args.wav_dir / f'{phrase.replace(" ", "_")}.wav'
            if not path.is_file():
                raise SystemExit(f'FAIL: {path} missing')
            return [('wav', read_wav(path))]
        if not (shutil.which('espeak-ng') and shutil.which('ffmpeg')):
            raise SystemExit('FAIL: synthetic clips need espeak-ng and ffmpeg, or pass --wav-dir')
        return [(v, read_wav(synthesise(phrase, work, v))) for v in voices]

    rng = np.random.default_rng(args.seed)
    levels = [float(v) for v in args.levels.split(',')]
    snrs = [float(v) for v in args.snrs.split(',')]

    detector = WhisperPhraseDetector(
        model_size=args.model, device='cpu', compute_type='int8',
        mic_channels=1, mic_beam_channel=0,
        adaptive_gate=not args.fixed_gate, cooldown_s=0.0,
        confidence_threshold=args.confidence, no_speech_threshold=args.no_speech)
    detector._start_audio_stream = lambda: None
    with contextlib.redirect_stdout(io.StringIO()):
        detector.setup()

    gate = 'FIXED 0.015' if args.fixed_gate else 'ADAPTIVE (margin 2.0)'
    print(f'gate: {gate}  model: {args.model}  conf>={args.confidence}  '
          f'no_speech<={args.no_speech}  reverb: {args.reverb_s}s  seed: {args.seed}')
    positives = clips_for(args.phrase)
    heard, total, stages, latencies = 0, 0, {}, []
    try:
        print(f'{"level":>7} {"snr":>5}  ' + '  '.join(f'{v:>10}' for v, _ in positives))
        for level in levels:
            for snr in snrs:
                row = []
                for voice, clean in positives:
                    buffer = io.StringIO()
                    with contextlib.redirect_stdout(buffer):
                        stage, hit = run_clip(detector, degrade(clean, level, snr, args.reverb_s, rng),
                                              buffer.getvalue)
                    total += 1
                    if stage == 'HEARD':
                        heard += 1
                        latencies.append(hit[1])
                    else:
                        stages[stage] = stages.get(stage, 0) + 1
                    row.append(stage)
                print(f'{level:>6.0f}dB {snr:>4.0f}  ' + '  '.join(f'{c:>10}' for c in row))

        false_triggers, negative_total = 0, 0
        for phrase in negatives:
            for voice, clean in clips_for(phrase):
                for level in (0.0, -12.0):
                    buffer = io.StringIO()
                    with contextlib.redirect_stdout(buffer):
                        stage, _ = run_clip(detector, degrade(clean, level, 30.0, args.reverb_s, rng),
                                            buffer.getvalue)
                    negative_total += 1
                    if stage == 'HEARD':
                        false_triggers += 1
                        print(f'FALSE TRIGGER: "{phrase}" voice={voice} level={level}dB')
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            detector.teardown()

    print(f'\n"{args.phrase}" recognised {heard}/{total} ({100.0 * heard / total:.0f}%)')
    if stages:
        print('misses by stage: ' + ', '.join(f'{k}={v}' for k, v in sorted(stages.items())))
    if latencies:
        print(f'speech end -> detection: median {np.median(latencies):.2f}s '
              f'max {max(latencies):.2f}s (this CPU, not the robot)')
    print(f'false triggers on {negatives}: {false_triggers}/{negative_total}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
