"""Phase 0 benchmark: Whisper inference latency on Jetson Orin NX.

Measures p50/p95/p99 inference time for tiny.en and base.en across
CUDA FP16, CUDA int8, and CPU int8 backends with 1.0s and 1.2s audio
windows. Results determine the ASR backend gate decision.

Gate A (pass):  tiny.en CUDA p95 <= 100ms  -> proceed with faster-whisper
Gate B (fail):  tiny.en CUDA p95 > 150ms   -> switch to whisper_trt
Gate C (middle): 100ms < p95 <= 150ms      -> relax target to 0.4s

Usage (on Jetson, source ROS2 first):
    python3 -u benchmark_whisper.py

    # With pre-recorded samples:
    python3 -u benchmark_whisper.py --samples-dir /path/to/wav/files

    # Specific configs only:
    python3 -u benchmark_whisper.py --models tiny.en --devices cuda
"""

import argparse
import pathlib
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Resolve local model paths
# ---------------------------------------------------------------------------
_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _resolve_model(model_size: str) -> str:
    """Find local faster-whisper model cache, or fall back to HF name."""
    local = _PKG_ROOT / "models" / f"faster-whisper-{model_size}"
    if not local.is_dir():
        return model_size
    snapshots = list(local.rglob("snapshots"))
    if snapshots:
        snap_dirs = list(snapshots[0].iterdir())
        if snap_dirs:
            return str(snap_dirs[0])
    return str(local)


# ---------------------------------------------------------------------------
# Audio sample generation / loading
# ---------------------------------------------------------------------------

def _generate_synthetic_samples(n: int, duration_s: float, sr: int = 16000) -> list[np.ndarray]:
    """Generate synthetic 'come here'-like audio samples (voiced noise bursts).

    These are NOT real speech -- they exercise the inference pipeline at the
    right input shape. For accuracy testing, use real recorded samples.
    """
    samples = []
    for _ in range(n):
        # Simulate voiced region (~0.7s) with silence padding
        voiced_len = int(0.7 * sr)
        total_len = int(duration_s * sr)
        offset = np.random.randint(0, max(1, total_len - voiced_len))
        audio = np.zeros(total_len, dtype=np.float32)
        t = np.arange(voiced_len, dtype=np.float32) / sr
        voiced = 0.3 * np.sin(2 * np.pi * 200 * t) * np.random.uniform(0.5, 1.0)
        voiced += np.random.normal(0, 0.05, voiced_len).astype(np.float32)
        audio[offset:offset + voiced_len] = voiced
        samples.append(audio)
    return samples


def _load_wav_samples(samples_dir: str, duration_s: float, sr: int = 16000) -> list[np.ndarray]:
    """Load WAV files from a directory, trim/pad to target duration."""
    import soundfile as sf
    samples = []
    target_len = int(duration_s * sr)
    p = pathlib.Path(samples_dir)
    for wav in sorted(p.glob("*.wav")):
        data, file_sr = sf.read(wav, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        if file_sr != sr:
            # Simple resample by linear interpolation
            indices = np.linspace(0, len(data) - 1, int(len(data) * sr / file_sr))
            data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
        # Pad or trim
        if len(data) < target_len:
            data = np.pad(data, (0, target_len - len(data)))
        else:
            data = data[:target_len]
        samples.append(data)
    return samples


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: int) -> float:
    s = sorted(values)
    idx = int(len(s) * p / 100)
    idx = min(idx, len(s) - 1)
    return s[idx]


def benchmark_config(
    model_size: str,
    device: str,
    compute_type: str,
    samples: list[np.ndarray],
    warmup: int = 3,
) -> dict:
    """Benchmark a single Whisper configuration. Returns timing stats."""
    from faster_whisper import WhisperModel

    model_path = _resolve_model(model_size)
    print(f"\n  Loading {model_size} on {device}/{compute_type} from {model_path}...")

    t0 = time.monotonic()
    model = WhisperModel(model_path, device=device, compute_type=compute_type)
    load_time = time.monotonic() - t0
    print(f"  Model loaded in {load_time:.2f}s")

    # Warmup runs (not counted)
    for i in range(min(warmup, len(samples))):
        segments, _ = model.transcribe(samples[i], beam_size=1, language="en")
        for _ in segments:
            pass

    # Timed runs
    times = []
    transcripts = []
    for sample in samples:
        t_start = time.monotonic()
        segments, info = model.transcribe(samples[0] if len(samples) == 1 else sample,
                                          beam_size=1, language="en")
        text_parts = []
        for seg in segments:
            text_parts.append(seg.text.strip())
        t_end = time.monotonic()
        times.append(t_end - t_start)
        transcripts.append(" ".join(text_parts))

    # GPU memory
    gpu_mem_mb = 0.0
    if device == "cuda":
        try:
            import torch
            gpu_mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
        except Exception:
            pass

    del model

    return {
        "model": model_size,
        "device": device,
        "compute_type": compute_type,
        "n_runs": len(times),
        "p50_ms": _percentile(times, 50) * 1000,
        "p95_ms": _percentile(times, 95) * 1000,
        "p99_ms": _percentile(times, 99) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "gpu_mem_mb": gpu_mem_mb,
        "load_time_s": load_time,
        "sample_transcript": transcripts[0] if transcripts else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Whisper inference on Jetson")
    parser.add_argument("--models", nargs="+", default=["tiny.en", "base.en"],
                        help="Model sizes to test")
    parser.add_argument("--devices", nargs="+", default=["cuda", "cpu"],
                        help="Devices to test")
    parser.add_argument("--windows", nargs="+", type=float, default=[1.0, 1.2],
                        help="Audio window durations in seconds")
    parser.add_argument("--n-runs", type=int, default=50,
                        help="Number of timed inference runs per config")
    parser.add_argument("--samples-dir", type=str, default=None,
                        help="Directory of WAV files to use as test samples")
    args = parser.parse_args()

    configs = []
    for device in args.devices:
        if device == "cuda":
            configs.append(("cuda", "float16"))
            configs.append(("cuda", "int8"))
        else:
            configs.append(("cpu", "int8"))

    print("=" * 65)
    print("  WHISPER INFERENCE BENCHMARK")
    print(f"  Models: {args.models}")
    print(f"  Configs: {configs}")
    print(f"  Windows: {args.windows}s")
    print(f"  Runs per config: {args.n_runs}")
    print(f"  Samples: {'from ' + args.samples_dir if args.samples_dir else 'synthetic'}")
    print("=" * 65)

    all_results = []

    for window_s in args.windows:
        print(f"\n--- Window: {window_s}s ---")

        if args.samples_dir:
            samples = _load_wav_samples(args.samples_dir, window_s)
            if not samples:
                print(f"  No WAV files found in {args.samples_dir}, using synthetic")
                samples = _generate_synthetic_samples(args.n_runs, window_s)
        else:
            samples = _generate_synthetic_samples(args.n_runs, window_s)

        # Ensure we have enough samples
        while len(samples) < args.n_runs:
            samples.extend(samples[:args.n_runs - len(samples)])
        samples = samples[:args.n_runs]

        for model_size in args.models:
            for device, compute_type in configs:
                label = f"{model_size} / {device} / {compute_type}"
                print(f"\n  [{label}]")
                try:
                    result = benchmark_config(
                        model_size, device, compute_type, samples
                    )
                    result["window_s"] = window_s
                    all_results.append(result)
                    print(f"    p50={result['p50_ms']:.1f}ms  "
                          f"p95={result['p95_ms']:.1f}ms  "
                          f"p99={result['p99_ms']:.1f}ms  "
                          f"min={result['min_ms']:.1f}ms  "
                          f"max={result['max_ms']:.1f}ms")
                    if result["gpu_mem_mb"] > 0:
                        print(f"    GPU mem: {result['gpu_mem_mb']:.1f} MB")
                    print(f"    Sample transcript: \"{result['sample_transcript'][:60]}\"")
                except Exception as e:
                    print(f"    FAILED: {e}")

    # Summary table
    print("\n")
    print("=" * 95)
    print("  RESULTS SUMMARY")
    print("=" * 95)
    print(f"  {'Model':<10} {'Device':<6} {'Type':<8} {'Win':<5} "
          f"{'p50':>7} {'p95':>7} {'p99':>7} {'max':>7} {'GPU MB':>7}  GATE")
    print("-" * 95)

    for r in all_results:
        p95 = r["p95_ms"]
        if "tiny" in r["model"] and r["device"] == "cuda":
            if p95 <= 100:
                gate = "PASS (Gate A)"
            elif p95 <= 150:
                gate = "MARGINAL (Gate C)"
            else:
                gate = "FAIL (Gate B -> whisper_trt)"
        else:
            gate = ""

        print(f"  {r['model']:<10} {r['device']:<6} {r['compute_type']:<8} "
              f"{r['window_s']:<5.1f} "
              f"{r['p50_ms']:>6.1f}ms {r['p95_ms']:>6.1f}ms "
              f"{r['p99_ms']:>6.1f}ms {r['max_ms']:>6.1f}ms "
              f"{r['gpu_mem_mb']:>6.1f}MB  {gate}")

    print()
    print("Gate A: tiny.en CUDA p95 <= 100ms -> proceed with faster-whisper")
    print("Gate B: tiny.en CUDA p95 > 150ms  -> switch to whisper_trt")
    print("Gate C: 100-150ms                 -> relax target to 0.4s")


if __name__ == "__main__":
    main()
