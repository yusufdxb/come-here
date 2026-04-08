#!/usr/bin/env python3
"""Record audio samples for fine-tuning Whisper on "come here" detection.

Records short clips from the default microphone and saves them as WAV files.
The script alternates between prompting for positive samples ("come here")
and negative samples (other speech, silence, noise).

Usage:
    python3 record_samples.py --output-dir data/
    python3 record_samples.py --output-dir data/ --duration 3.0 --positive-only
"""

import argparse
import os
import time
import wave

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16000
CHANNELS = 1


def record_clip(duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record a single audio clip."""
    print(f"  Recording for {duration:.1f}s...", end="", flush=True)
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()
    print(" done.")
    return audio.flatten()


def save_wav(path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Save float32 audio as 16-bit WAV."""
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Record samples for Whisper fine-tuning")
    parser.add_argument("--output-dir", default="data", help="Directory to save recordings")
    parser.add_argument("--duration", type=float, default=3.0, help="Clip duration in seconds")
    parser.add_argument("--num-positive", type=int, default=20, help="Number of positive samples")
    parser.add_argument("--num-negative", type=int, default=20, help="Number of negative samples")
    parser.add_argument("--positive-only", action="store_true", help="Only record positive samples")
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    args = parser.parse_args()

    if args.device is not None:
        sd.default.device = args.device

    pos_dir = os.path.join(args.output_dir, "positive")
    neg_dir = os.path.join(args.output_dir, "negative")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    # Count existing samples to avoid overwriting
    pos_count = len([f for f in os.listdir(pos_dir) if f.endswith(".wav")])
    neg_count = len([f for f in os.listdir(neg_dir) if f.endswith(".wav")])

    print(f"Audio device: {sd.query_devices(sd.default.device[0])['name']}")
    print(f"Existing samples: {pos_count} positive, {neg_count} negative")
    print(f"Will record: {args.num_positive} positive" +
          (f", {args.num_negative} negative" if not args.positive_only else ""))
    print()

    # Positive samples
    print("=== POSITIVE SAMPLES ===")
    print('Say "come here" clearly after each prompt.\n')
    for i in range(args.num_positive):
        input(f"  [{pos_count + i + 1}] Press Enter, then say 'come here'... ")
        audio = record_clip(args.duration)
        path = os.path.join(pos_dir, f"come_here_{pos_count + i:04d}.wav")
        save_wav(path, audio)
        print(f"  Saved: {path}")

    if args.positive_only:
        print(f"\nDone. Recorded {args.num_positive} positive samples.")
        return

    # Negative samples
    print("\n=== NEGATIVE SAMPLES ===")
    print("Say anything EXCEPT 'come here', or make background noise.\n")
    print("Ideas: random words, silence, clapping, other commands, conversations.\n")
    for i in range(args.num_negative):
        input(f"  [{neg_count + i + 1}] Press Enter, then speak/make noise... ")
        audio = record_clip(args.duration)
        path = os.path.join(neg_dir, f"negative_{neg_count + i:04d}.wav")
        save_wav(path, audio)
        print(f"  Saved: {path}")

    print(f"\nDone. Recorded {args.num_positive} positive + {args.num_negative} negative samples.")
    print(f"Total in {args.output_dir}: {pos_count + args.num_positive} positive, "
          f"{neg_count + args.num_negative} negative")


if __name__ == "__main__":
    main()
