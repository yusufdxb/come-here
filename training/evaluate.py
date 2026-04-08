#!/usr/bin/env python3
"""Evaluate fine-tuned Whisper model on held-out samples or live mic.

Tests:
  1. Accuracy on recorded positive/negative samples
  2. (Optional) Live mic test -- speak and see real-time detection

Usage:
    # Evaluate on recorded data
    python3 evaluate.py --adapter-dir output/lora_adapter/ --data-dir data/

    # Live mic test
    python3 evaluate.py --adapter-dir output/lora_adapter/ --live

    # Compare base vs fine-tuned
    python3 evaluate.py --data-dir data/ --compare
"""

import argparse
import os
import time

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

import soundfile as sf


TRIGGER_PHRASES = {"come here", "come over here"}


def load_model(base_model: str, adapter_dir: str = None):
    """Load base Whisper model, optionally with LoRA adapter."""
    processor = WhisperProcessor.from_pretrained(base_model)

    if adapter_dir:
        model = WhisperForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
        print(f"Loaded fine-tuned model from {adapter_dir}")
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=torch.float16
        )
        print(f"Loaded base model: {base_model}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, processor, device


def transcribe_file(model, processor, device, wav_path: str) -> tuple[str, float]:
    """Transcribe a WAV file. Returns (text, inference_time_ms)."""
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    inputs = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device, dtype=torch.float16)

    t0 = time.time()
    with torch.no_grad():
        predicted_ids = model.generate(inputs, max_new_tokens=30)
    elapsed_ms = (time.time() - t0) * 1000

    text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text.strip().lower(), elapsed_ms


def evaluate_dataset(model, processor, device, data_dir: str):
    """Run evaluation on positive and negative samples."""
    results = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "times": []}

    pos_dir = os.path.join(data_dir, "positive")
    neg_dir = os.path.join(data_dir, "negative")

    # Positive samples
    if os.path.isdir(pos_dir):
        wavs = sorted(f for f in os.listdir(pos_dir) if f.endswith(".wav"))
        print(f"\nPositive samples ({len(wavs)}):")
        for wav in wavs:
            path = os.path.join(pos_dir, wav)
            text, ms = transcribe_file(model, processor, device, path)
            is_match = any(t in text for t in TRIGGER_PHRASES)
            results["tp" if is_match else "fn"] += 1
            results["times"].append(ms)
            status = "HIT" if is_match else "MISS"
            print(f"  [{status}] {wav}: \"{text}\" ({ms:.0f}ms)")

    # Negative samples
    if os.path.isdir(neg_dir):
        wavs = sorted(f for f in os.listdir(neg_dir) if f.endswith(".wav"))
        print(f"\nNegative samples ({len(wavs)}):")
        for wav in wavs:
            path = os.path.join(neg_dir, wav)
            text, ms = transcribe_file(model, processor, device, path)
            is_match = any(t in text for t in TRIGGER_PHRASES)
            results["fp" if is_match else "tn"] += 1
            results["times"].append(ms)
            status = "FALSE+" if is_match else "OK"
            print(f"  [{status}] {wav}: \"{text}\" ({ms:.0f}ms)")

    # Summary
    tp, fp, tn, fn = results["tp"], results["fp"], results["tn"], results["fn"]
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_ms = np.mean(results["times"]) if results["times"] else 0

    print(f"\n{'='*50}")
    print(f"Results: {total} samples")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1:        {f1:.1%}")
    print(f"  Avg latency: {avg_ms:.0f}ms")
    print(f"{'='*50}")

    return results


def live_test(model, processor, device, duration: float = 3.0):
    """Real-time mic test loop."""
    import sounddevice as sd

    print(f"\nLive test (Ctrl+C to stop)")
    print(f"Recording {duration:.1f}s chunks from default mic...\n")

    while True:
        try:
            audio = sd.rec(
                int(duration * 16000), samplerate=16000, channels=1, dtype="float32"
            )
            sd.wait()
            audio = audio.flatten()

            if np.max(np.abs(audio)) < 0.01:
                print("  [silence]")
                continue

            inputs = processor.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device, dtype=torch.float16)

            t0 = time.time()
            with torch.no_grad():
                predicted_ids = model.generate(inputs, max_new_tokens=30)
            ms = (time.time() - t0) * 1000

            text = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip().lower()

            is_trigger = any(t in text for t in TRIGGER_PHRASES)
            marker = ">>> TRIGGERED <<<" if is_trigger else ""
            print(f'  "{text}" ({ms:.0f}ms) {marker}')

        except KeyboardInterrupt:
            print("\nStopped.")
            break


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper fine-tune")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_model = os.path.join(script_dir, "..", "models", "whisper-base.en")
    default_model = local_model if os.path.isdir(local_model) else "openai/whisper-base.en"
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--adapter-dir", default=None, help="LoRA adapter directory")
    parser.add_argument("--data-dir", default="data", help="Test data directory")
    parser.add_argument("--live", action="store_true", help="Live mic test")
    parser.add_argument("--compare", action="store_true",
                        help="Compare base vs fine-tuned on same data")
    args = parser.parse_args()

    if args.compare:
        print("=== BASE MODEL ===")
        base_model, base_proc, device = load_model(args.model)
        base_results = evaluate_dataset(base_model, base_proc, device, args.data_dir)

        if args.adapter_dir:
            print("\n\n=== FINE-TUNED MODEL ===")
            ft_model, ft_proc, device = load_model(args.model, args.adapter_dir)
            ft_results = evaluate_dataset(ft_model, ft_proc, device, args.data_dir)
        return

    model, processor, device = load_model(args.model, args.adapter_dir)

    if args.live:
        live_test(model, processor, device)
    else:
        evaluate_dataset(model, processor, device, args.data_dir)


if __name__ == "__main__":
    main()
