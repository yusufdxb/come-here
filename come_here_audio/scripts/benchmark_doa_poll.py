"""Phase 0 benchmark: ReSpeaker DOA USB HID polling overhead.

Measures USB control transfer round-trip time and determines the safe
maximum polling rate for continuous DOA tracking.

Usage (on Jetson):
    python3 -u benchmark_doa_poll.py

Output:
    - USB HID round-trip p50/p95/max
    - Recommended polling rate
    - Data freshness test at candidate rates
"""

import struct
import sys
import time

import usb.core
import usb.util

VENDOR_ID = 0x2886
PRODUCT_ID = 0x0018
CTRL_TIMEOUT = 100000

REG_DOAANGLE = (21, 0)
REG_VOICEACTIVITY = (19, 32)


def read_register(dev, reg_id: int, offset: int) -> int:
    cmd = 0x80 | offset | 0x40
    response = dev.ctrl_transfer(
        usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
        0, cmd, reg_id, 8, CTRL_TIMEOUT,
    )
    return struct.unpack(b'ii', response.tobytes())[0]


def percentile(values: list[float], p: int) -> float:
    s = sorted(values)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


def benchmark_roundtrip(dev, n_runs: int = 200) -> dict:
    """Measure raw USB HID control transfer round-trip time."""
    times = []
    for _ in range(n_runs):
        t0 = time.monotonic()
        read_register(dev, *REG_DOAANGLE)
        read_register(dev, *REG_VOICEACTIVITY)
        t1 = time.monotonic()
        times.append(t1 - t0)

    return {
        "n_runs": n_runs,
        "p50_us": percentile(times, 50) * 1e6,
        "p95_us": percentile(times, 95) * 1e6,
        "p99_us": percentile(times, 99) * 1e6,
        "max_us": max(times) * 1e6,
        "min_us": min(times) * 1e6,
    }


def benchmark_poll_rate(dev, target_hz: float, duration_s: float = 5.0) -> dict:
    """Sustained polling at a target rate. Measures achieved rate and jitter."""
    interval = 1.0 / target_hz
    samples = []
    overruns = 0

    start = time.monotonic()
    next_tick = start
    while time.monotonic() - start < duration_s:
        t0 = time.monotonic()
        doa = read_register(dev, *REG_DOAANGLE)
        vad = read_register(dev, *REG_VOICEACTIVITY)
        t1 = time.monotonic()
        samples.append((t1, doa, vad, t1 - t0))

        next_tick += interval
        sleep_time = next_tick - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            overruns += 1

    actual_hz = len(samples) / duration_s
    poll_times = [s[3] for s in samples]

    # Check DOA value changes (data freshness)
    doa_values = [s[1] for s in samples]
    unique_doa = len(set(doa_values))
    doa_changes = sum(1 for i in range(1, len(doa_values)) if doa_values[i] != doa_values[i - 1])

    return {
        "target_hz": target_hz,
        "actual_hz": actual_hz,
        "total_samples": len(samples),
        "overruns": overruns,
        "poll_p50_us": percentile(poll_times, 50) * 1e6,
        "poll_p95_us": percentile(poll_times, 95) * 1e6,
        "unique_doa_values": unique_doa,
        "doa_changes": doa_changes,
    }


def main():
    print("Connecting to ReSpeaker Mic Array v2.0...")
    dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
    if dev is None:
        print("ERROR: ReSpeaker not found!")
        sys.exit(1)
    print("Found.\n")

    # 1. Raw round-trip benchmark
    print("=" * 55)
    print("  USB HID ROUND-TRIP BENCHMARK (DOA + VAD reads)")
    print("=" * 55)

    rt = benchmark_roundtrip(dev, n_runs=200)
    print(f"  Runs:   {rt['n_runs']}")
    print(f"  p50:    {rt['p50_us']:.0f} us")
    print(f"  p95:    {rt['p95_us']:.0f} us")
    print(f"  p99:    {rt['p99_us']:.0f} us")
    print(f"  max:    {rt['max_us']:.0f} us")
    print(f"  min:    {rt['min_us']:.0f} us")

    # Recommend: poll period > 2x p95 round-trip
    safe_period_us = rt["p95_us"] * 2
    max_safe_hz = 1e6 / safe_period_us
    print(f"\n  Safe max polling rate: {max_safe_hz:.0f} Hz "
          f"(2x p95 = {safe_period_us:.0f} us period)")

    # 2. Sustained polling at candidate rates
    print()
    print("=" * 55)
    print("  SUSTAINED POLLING BENCHMARK (5s per rate)")
    print("=" * 55)

    candidate_rates = [20, 30, 50, 75, 100]
    print(f"\n  {'Rate':>6} {'Actual':>8} {'Overruns':>9} "
          f"{'Poll p95':>10} {'DOA changes':>12} {'Unique DOA':>11}")
    print("-" * 65)

    for rate in candidate_rates:
        r = benchmark_poll_rate(dev, target_hz=rate, duration_s=5.0)
        print(f"  {r['target_hz']:>5}Hz {r['actual_hz']:>7.1f}Hz "
              f"{r['overruns']:>8}  "
              f"{r['poll_p95_us']:>9.0f}us "
              f"{r['doa_changes']:>11} "
              f"{r['unique_doa_values']:>10}")

    print()
    print("INTERPRETATION:")
    print("  - If overruns > 0 at a rate, that rate is too fast for reliable polling.")
    print("  - DOA changes indicate data freshness (0 = register not updating).")
    print("  - Pick the highest rate with 0 overruns and meaningful DOA changes.")
    print(f"  - Recommended initial rate: {min(max_safe_hz, 50):.0f} Hz")
    print()
    print("Use the chosen rate as poll_rate_hz in ReSpeakerDOAProvider.start_continuous().")


if __name__ == "__main__":
    main()
