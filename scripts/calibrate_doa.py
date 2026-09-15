#!/usr/bin/env python3
"""Center the ReSpeaker DOA on GO2 forward. No ROS launch, no robot motion.

Stop the come-here launch first: ALSA gives the capture device to one process.

  python3 scripts/calibrate_doa.py            # calibrate: AHEAD phase, then LEFT phase
  python3 scripts/calibrate_doa.py --verify   # front / 45L / 45R / 90L / 90R check
  python3 scripts/calibrate_doa.py --selftest # synthetic, no microphone

Calibrate (robot standing still, quiet room, caller about 1.5 m away):
  1. AHEAD: stand directly in front of the robot's nose, press Enter, then say
     "come here" (or clap) repeatedly, about 2 s apart, until it has enough.
  2. LEFT: stand on the robot's LEFT side (90 deg), press Enter, speak again.
     This decides whether the array is mirrored. Mirror is decided BEFORE the
     offset, because robot = (mirror ? -array : array) + offset.
  3. The result is saved to ~/come_here_trials/doa_calibration.json, which
     audio_node loads at startup (launch doa_offset_deg / doa_mirror override).

Circular statistics throughout: bearings are averaged as unit vectors, never
as plain numbers, so 359 deg and 1 deg average to 0 deg, not 180 deg.
"""

import argparse
import datetime
import json
import math
import os
import socket
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
try:
    import come_here_audio.srp_doa  # noqa: F401
except ImportError:  # running from a source checkout without the install sourced
    sys.path.insert(0, os.path.join(_HERE, '..', 'come_here_audio'))

from doa_probe import (  # noqa: E402  same segmentation as the lab probe
    FRAME, GATE_FLOOR, GATE_MARGIN, MAX_UTT_S, MIN_UTT_S, RATE, SILENCE_END_S,
)

DEFAULT_OUT = os.path.expanduser('~/come_here_trials/doa_calibration.json')
OUTLIER_DEG = 30.0          # samples this far from the circular mean are dropped
MAX_AHEAD_STD_DEG = 15.0    # a wider spread means the room or the gate is not usable
VERIFY_TOL_DEG = 20.0


def wrap_deg(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def circular_stats(angles_deg):
    """(mean_deg, resultant_length R in 0..1, circular std deg, n)."""
    a = np.radians(np.asarray(angles_deg, dtype=np.float64))
    if a.size == 0:
        return None, 0.0, float('inf'), 0
    s, c = float(np.mean(np.sin(a))), float(np.mean(np.cos(a)))
    r = math.hypot(s, c)
    mean = math.degrees(math.atan2(s, c))
    std = math.degrees(math.sqrt(-2.0 * math.log(r))) if r > 1e-9 else float('inf')
    return mean, r, std, int(a.size)


def robust_circular_mean(angles_deg, outlier_deg=OUTLIER_DEG):
    """Circular mean after dropping samples more than outlier_deg from the first mean."""
    mean, _, _, _ = circular_stats(angles_deg)
    if mean is None:
        return None, 0.0, float('inf'), 0, []
    kept = [a for a in angles_deg if abs(wrap_deg(a - mean)) <= outlier_deg]
    dropped = [a for a in angles_deg if abs(wrap_deg(a - mean)) > outlier_deg]
    if len(kept) < max(2, len(angles_deg) // 2):   # no majority cluster: report as is
        m, r, std, n = circular_stats(angles_deg)
        return m, r, std, n, []
    m, r, std, n = circular_stats(kept)
    return m, r, std, n, dropped


def corrected_deg(array_deg: float, offset_deg: float, mirror: bool) -> float:
    """Same math as come_here_audio.srp_doa.to_robot_frame, in degrees."""
    return wrap_deg((-array_deg if mirror else array_deg) + offset_deg)


def solve(ahead_array_deg: float, left_array_deg=None, mirror_override=None):
    """Mirror first (from the LEFT phase), then the offset that puts AHEAD at 0.

    Returns (offset_deg, mirror, note). Raises ValueError when the LEFT phase
    does not look like a quarter turn from AHEAD in either direction.
    """
    if mirror_override is not None:
        mirror, note = bool(mirror_override), 'mirror forced by --mirror'
    elif left_array_deg is None:
        raise ValueError('LEFT phase missing: cannot tell mirror; rerun or pass --mirror true|false')
    else:
        delta = wrap_deg(left_array_deg - ahead_array_deg)
        if 45.0 <= delta <= 135.0:
            mirror = False
        elif -135.0 <= delta <= -45.0:
            mirror = True
        else:
            raise ValueError(f'LEFT is {delta:+.0f} deg from AHEAD in the array frame; '
                             'expected about +90 or -90. Caller not at the side, or DOA unusable.')
        note = f'LEFT - AHEAD = {delta:+.1f} deg in the array frame'
    offset = wrap_deg(-(-ahead_array_deg if mirror else ahead_array_deg))
    return offset, mirror, note


class Listener:
    """Continuous 6-channel capture; yields one voiced utterance at a time."""

    def __init__(self, device: str):
        import sounddevice as sd
        from come_here_audio.mic_select import resolve
        index, name, channels, far_field = resolve(device, True)
        if index is None or not far_field:
            raise SystemExit(f'need the 6-channel ReSpeaker, got {name} ({channels}ch)')
        self.name = f'{index}:{name} ({channels}ch)'
        self._pending = []
        self._stream = sd.InputStream(samplerate=RATE, channels=6, dtype='float32', device=index,
                                      blocksize=int(0.1 * RATE), callback=self._cb)
        self._levels = []
        self._stream.start()

    def _cb(self, indata, frames, t, status):
        self._pending.append(indata.copy())

    def flush(self):
        self._pending = []

    def next_utterance(self, timeout_s: float):
        frame_n = int(FRAME * RATE)
        buf = np.zeros((0, 6), dtype=np.float32)
        utt, silent = [], 0
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            time.sleep(0.05)
            chunks, self._pending = self._pending, []
            if chunks:
                buf = np.concatenate([buf] + chunks)
            while len(buf) >= frame_n:
                frame, buf = buf[:frame_n], buf[frame_n:]
                rms = float(np.sqrt(np.mean(frame[:, 0] ** 2)))
                self._levels = (self._levels + [rms])[-int(3.0 / FRAME):]
                floor = float(np.percentile(self._levels, 25)) if len(self._levels) >= 10 else rms
                voiced = rms >= max(GATE_FLOOR, floor * GATE_MARGIN)
                if not utt:
                    if voiced:
                        utt, silent = [frame], 0
                    continue
                utt.append(frame)
                silent = 0 if voiced else silent + 1
                length_s = len(utt) * FRAME
                if silent * FRAME >= SILENCE_END_S or length_s >= MAX_UTT_S:
                    spoken_s = length_s - silent * FRAME
                    block, utt = np.concatenate(utt), []
                    if spoken_s >= MIN_UTT_S:
                        return block
        return None

    def close(self):
        self._stream.stop()
        self._stream.close()


def collect(listener, doa, label, want, min_conf, timeout_s, cal=None, firmware=None):
    """Prompt, then gather `want` estimates with confidence >= min_conf. Returns array degrees."""
    from come_here_audio.srp_doa import RESPEAKER_V2_RAW_CHANNELS
    input(f'\n>>> {label}: press Enter, then say "come here" every ~2 s ({want} needed) ')
    listener.flush()
    got, start = [], time.monotonic()
    while len(got) < want and time.monotonic() - start < timeout_s:
        block = listener.next_utterance(timeout_s - (time.monotonic() - start))
        if block is None:
            break
        est = doa.estimate(block[:, list(RESPEAKER_V2_RAW_CHANNELS)])
        fw = ''
        if firmware is not None:
            try:
                from come_here_audio import respeaker_tune
                fw = f'  firmware {int(respeaker_tune.read_parameter(firmware, "DOAANGLE")):3d}'
            except Exception:  # noqa: BLE001
                fw = ''
        if est is None:
            print(f'  {len(block) / RATE:4.2f}s  no estimate (too short or silent)')
            continue
        raw = math.degrees(est.azimuth_rad)
        keep = est.confidence >= min_conf
        line = f'  RAW DOA: {raw:+6.1f} deg  conf {est.confidence:.2f}  peak {est.peak:.3f}'
        if cal is not None:
            line += (f'  OFFSET: {cal["offset_deg"]:+6.1f}  MIRROR: {cal["mirror"]}'
                     f'  CORRECTED: {corrected_deg(raw, cal["offset_deg"], cal["mirror"]):+6.1f} deg')
        print(line + fw + ('' if keep else f'  (rejected: conf < {min_conf})'))
        if keep:
            got.append(raw)
    return got


def load_calibration(path):
    with open(path) as f:
        cal = json.load(f)
    return {'offset_deg': float(cal['offset_deg']), 'mirror': bool(cal['mirror'])}


def selftest() -> int:
    """Synthetic plane waves through the real estimator, with a known mount."""
    from come_here_audio.srp_doa import SrpPhatDoa, synthesize_plane_wave
    doa = SrpPhatDoa(sample_rate=RATE)
    rng = np.random.default_rng(1)
    n = RATE
    x = rng.normal(0.0, 1.0, n)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / RATE)
    spec[(f < 200) | (f > 3500)] = 0
    x = 0.05 * np.fft.irfft(spec, n=n)
    ok = True
    # wrap sanity: 359 and 1 average to 0
    m, _, _, _ = circular_stats([359.0, 1.0, 358.0, 2.0])
    print(f'circular mean of 359,1,358,2 = {m:+.2f} deg (plain mean would be 180)')
    ok &= abs(wrap_deg(m)) < 0.5
    for true_offset, true_mirror in ((29.0, False), (-150.0, True), (178.0, False), (0.0, True)):
        def array_for(robot_deg):
            a = robot_deg - true_offset
            return wrap_deg(-a if true_mirror else a)

        def measure(robot_deg, k=6):
            out = []
            for _ in range(k):
                az = math.radians(array_for(robot_deg) + rng.normal(0.0, 3.0))
                est = doa.estimate(synthesize_plane_wave(x, az, RATE))
                out.append(math.degrees(est.azimuth_rad))
            return out
        ahead = measure(0.0) + [array_for(0.0) + 120.0]      # one wild outlier
        ahead_mean, _, ahead_std, _, dropped = robust_circular_mean(ahead)
        left_mean = robust_circular_mean(measure(90.0, 3))[0]
        offset, mirror, _ = solve(ahead_mean, left_mean)
        worst = 0.0
        for robot in (0.0, 45.0, -45.0, 90.0, -90.0):
            got = circular_stats([corrected_deg(a, offset, mirror) for a in measure(robot, 3)])[0]
            worst = max(worst, abs(wrap_deg(got - robot)))
        good = mirror == true_mirror and worst <= 6.0 and len(dropped) == 1
        ok &= good
        print(f'mount offset {true_offset:+6.1f} mirror {true_mirror!s:5}: solved offset {offset:+6.1f} '
              f'mirror {mirror!s:5} ahead std {ahead_std:4.1f} dropped {len(dropped)} '
              f'worst verify error {worst:4.1f} deg -> {"ok" if good else "FAIL"}')
    print(f'selftest {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--verify', action='store_true', help='check front/45/90 with the saved calibration')
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--out', default=DEFAULT_OUT)
    p.add_argument('--device', default='ReSpeaker')
    p.add_argument('--ahead-count', type=int, default=8)
    p.add_argument('--left-count', type=int, default=4)
    p.add_argument('--verify-count', type=int, default=3)
    p.add_argument('--min-conf', type=float, default=0.3,
                   help='estimates below this confidence are not used (node threshold is separate)')
    p.add_argument('--mirror', choices=('auto', 'true', 'false'), default='auto')
    p.add_argument('--phase-timeout', type=float, default=60.0)
    args = p.parse_args(argv)

    if args.selftest:
        return selftest()

    from come_here_audio.srp_doa import SrpPhatDoa
    doa = SrpPhatDoa(sample_rate=RATE)
    firmware = None
    try:
        from come_here_audio import respeaker_tune
        firmware = respeaker_tune.find_device()
    except Exception:  # noqa: BLE001
        firmware = None
    listener = Listener(args.device)
    print(f'mic {listener.name}; robot must be standing still; nobody else talking')
    try:
        if args.verify:
            return verify(listener, doa, args, firmware)
        return calibrate(listener, doa, args, firmware)
    except KeyboardInterrupt:
        print('\ninterrupted, nothing saved')
        return 130
    finally:
        listener.close()


def calibrate(listener, doa, args, firmware) -> int:
    ahead = collect(listener, doa, 'AHEAD (stand directly in front of the nose)', args.ahead_count,
                    args.min_conf, args.phase_timeout, firmware=firmware)
    if len(ahead) < max(4, args.ahead_count // 2):
        print(f'FAIL: only {len(ahead)} usable AHEAD estimates; speak louder/closer or lower --min-conf')
        return 1
    a_mean, a_r, a_std, a_n, dropped = robust_circular_mean(ahead)
    print(f'AHEAD: circular mean {a_mean:+.1f} deg, circ std {a_std:.1f} deg, R {a_r:.2f}, '
          f'n {a_n}, outliers dropped {len(dropped)} {["%+.0f" % d for d in dropped]}')
    if a_std > MAX_AHEAD_STD_DEG:
        print(f'FAIL: AHEAD spread {a_std:.1f} deg > {MAX_AHEAD_STD_DEG}; DOA is not stable enough to calibrate')
        return 1

    left_mean, l_std, l_n = None, None, 0
    forced = None if args.mirror == 'auto' else (args.mirror == 'true')
    if forced is None:
        left = collect(listener, doa, "LEFT (stand at the robot's LEFT side, 90 deg)", args.left_count,
                       args.min_conf, args.phase_timeout, firmware=firmware)
        if len(left) < 2:
            print('FAIL: not enough LEFT estimates to decide mirror')
            return 1
        left_mean, _, l_std, l_n, _ = robust_circular_mean(left)
        print(f'LEFT: circular mean {left_mean:+.1f} deg, circ std {l_std:.1f} deg, n {l_n}')
    try:
        offset, mirror, note = solve(a_mean, left_mean, forced)
    except ValueError as exc:
        print(f'FAIL: {exc}')
        return 1

    residual = circular_stats([corrected_deg(a, offset, mirror) for a in ahead])
    print('\n' + '=' * 60)
    print(f'RAW DOA:   {a_mean:+6.1f} deg   (caller ahead, array frame)')
    print(f'MIRROR:    {mirror}   ({note})')
    print(f'OFFSET:    {offset:+6.1f} deg')
    print(f'CORRECTED: {corrected_deg(a_mean, offset, mirror):+6.1f} deg   '
          f'(ahead samples: mean {residual[0]:+.1f}, std {residual[2]:.1f})')
    if left_mean is not None:
        print(f'LEFT CORRECTED: {corrected_deg(left_mean, offset, mirror):+6.1f} deg (expect about +90)')
    print('=' * 60)

    cal = {
        'offset_deg': round(offset, 2), 'mirror': mirror,
        'ahead_array_deg': round(a_mean, 2), 'ahead_circ_std_deg': round(a_std, 2), 'ahead_n': a_n,
        'left_array_deg': None if left_mean is None else round(left_mean, 2), 'left_n': l_n,
        'min_conf': args.min_conf,
        'measured_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'host': socket.gethostname(), 'tool': 'scripts/calibrate_doa.py',
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(cal, f, indent=2)
    print(f'saved {args.out}')
    print(f'equivalent launch args: doa_offset_deg:={offset:.1f} doa_mirror:={str(mirror).lower()}')
    print('next: python3 scripts/calibrate_doa.py --verify')
    return 0


def verify(listener, doa, args, firmware) -> int:
    try:
        cal = load_calibration(args.out)
    except (OSError, KeyError, ValueError) as exc:
        print(f'no calibration at {args.out}: {exc}')
        return 1
    print(f'calibration {args.out}: offset {cal["offset_deg"]:+.1f} deg mirror {cal["mirror"]}')
    positions = (('FRONT', 0.0), ('45 deg LEFT', 45.0), ('45 deg RIGHT', -45.0),
                 ('90 deg LEFT', 90.0), ('90 deg RIGHT', -90.0))
    rows, all_ok = [], True
    for label, expect in positions:
        got = collect(listener, doa, f'{label} (expect {expect:+.0f})', args.verify_count,
                      args.min_conf, args.phase_timeout, cal=cal, firmware=firmware)
        if not got:
            rows.append((label, expect, None, None, False))
            all_ok = False
            continue
        mean = circular_stats([corrected_deg(a, cal['offset_deg'], cal['mirror']) for a in got])[0]
        err = abs(wrap_deg(mean - expect))
        ok = err <= VERIFY_TOL_DEG
        all_ok &= ok
        rows.append((label, expect, mean, err, ok))
    print('\n| Position | Expected | Corrected mean | Error | Result |')
    print('|---|---|---|---|---|')
    for label, expect, mean, err, ok in rows:
        m = '-' if mean is None else f'{mean:+.1f}'
        e = '-' if err is None else f'{err:.1f}'
        print(f'| {label} | {expect:+.0f} | {m} | {e} | {"PASS" if ok else "FAIL"} |')
    print(f'\nVERIFY {"PASS" if all_ok else "FAIL"} (tolerance {VERIFY_TOL_DEG:.0f} deg)')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
