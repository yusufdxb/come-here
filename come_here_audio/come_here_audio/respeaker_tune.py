"""Read and set the ReSpeaker Mic Array v2.0 DSP registers (XMOS XVF-3000).

The array runs beamforming, AEC, noise suppression, AGC and VAD in firmware.
Seeed rate it for voice pickup at up to 5 m, while come-here measured about
1 m on the robot. The firmware settings, AGC above all, decide how loud a
distant talker is by the time the software gate sees the signal.

    python3 -m come_here_audio.respeaker_tune                  # report registers
    python3 -m come_here_audio.respeaker_tune --profile far_field
    python3 -m come_here_audio.respeaker_tune --restore-defaults

Registers are volatile: they return to the firmware defaults on a power cycle.
audio_node applies the profile at startup when respeaker_profile is far_field.

Parameter ids, offsets, types and defaults follow the XVF-3000 tuning table
published with respeaker/usb_4_mic_array. AGCMAXGAIN is offset 1 and a float;
offset 4 is AGCTIME. Writing a float parameter with the integer wire format
does not fail, it writes nonsense, so the two formats are kept separate.
"""

import argparse
import struct
import sys

# name -> (id, offset, kind, default, description)
PARAMETERS = {
    'AGCONOFF': (19, 0, int, 1, 'automatic gain control, 0 off 1 on'),
    'AGCMAXGAIN': (19, 1, float, 31.6, 'max AGC gain factor, 31.6 = 30 dB, 1000 = 60 dB'),
    'AGCDESIREDLEVEL': (19, 2, float, 0.005, 'target output power, 0.005 = -23 dBov'),
    'AGCGAIN': (19, 3, float, 1.0, 'current AGC gain factor'),
    'AGCTIME': (19, 4, float, 0.5, 'AGC ramp time constant, seconds'),
    'STATNOISEONOFF': (19, 8, int, 1, 'stationary noise suppression, 0 off 1 on'),
    'GAMMA_NS': (19, 9, float, 1.0, 'over-subtraction factor for stationary noise'),
    'MIN_NS': (19, 10, float, 0.15, 'gain floor for noise suppression'),
    'GAMMAVAD_SR': (19, 39, float, 3.5, 'VAD threshold in dB (SDK tuning.py: default 3.5 dB); lower hears further'),
    'VOICEACTIVITY': (19, 32, int, 0, 'read-only: firmware VAD right now'),
    'SPEECHDETECTED': (19, 22, int, 0, 'read-only: firmware speech detection status'),
    'HPFONOFF': (18, 27, int, 0, 'high-pass filter 0 off, 1 70Hz, 2 125Hz, 3 180Hz'),
    'DOAANGLE': (21, 0, int, 0, 'read-only: direction of arrival, degrees'),
}

# AGCGAIN is 'rw' in the SDK table (respeaker/usb_4_mic_array tuning.py): the
# current gain can be seeded, and the AGC keeps riding it from there.
READ_ONLY = ('VOICEACTIVITY', 'SPEECHDETECTED', 'DOAANGLE')

# For a caller a few metres in front of the robot:
#   AGCONOFF 1         the DSP's own gain ride is the cheapest far-field win
#   AGCMAXGAIN 1000    allow the full 60 dB instead of the 30 dB default
#   AGCDESIREDLEVEL    a hotter target than -23 dBov, so a distant talker lands
#                      well above the software gate
#   STATNOISEONOFF 1   keep stationary suppression: the robot hums
#   GAMMA_NS 1.0       but do not over-subtract, which eats quiet consonants
#   MIN_NS 0.15        leave a gain floor so suppressed frames are not silence
#   HPFONOFF 1         70 Hz, below voice and above most chassis rumble
#   GAMMAVAD_SR 2.0    the SDK (usb_4_mic_array/tuning.py) documents this register
#                      in dB, default 3.5 dB; 2.0 dB is ODIN's far-field value and
#                      makes the firmware VAD that corroborates DOA hear further
#   AGCGAIN 10         seed the CURRENT gain. Lab 2026-09-15: after a power
#                      loss the array sat at 1.25-1.75 for 10+ min and ch0 was
#                      0.46x the raw capsules (09-14 working session: 3.9x); a
#                      2.5 m "come here" never opened the gate. Seeded at 10 it
#                      held for 20 s and ch0 came back to 3.7x. Every launch
#                      now starts from that state instead of a cold AGC.
FAR_FIELD_PROFILE = {
    'GAMMAVAD_SR': 2.0,
    'AGCONOFF': 1,
    'AGCMAXGAIN': 1000.0,
    'AGCDESIREDLEVEL': 0.03,
    'AGCGAIN': 10.0,
    'STATNOISEONOFF': 1,
    'GAMMA_NS': 1.0,
    'MIN_NS': 0.15,
    'HPFONOFF': 1,
}

PROFILES = {'far_field': FAR_FIELD_PROFILE}

_VENDOR_ID = 0x2886
_PRODUCT_ID = 0x0018
_CTRL_TIMEOUT = 100000


def find_device():
    import usb.core
    return usb.core.find(idVendor=_VENDOR_ID, idProduct=_PRODUCT_ID)


def read_parameter(device, name: str):
    import usb.util
    parameter_id, offset, kind, _default, _doc = PARAMETERS[name]
    command = 0x80 | offset
    if kind is int:
        command |= 0x40
    response = device.ctrl_transfer(
        usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
        0, command, parameter_id, 8, _CTRL_TIMEOUT)
    first, second = struct.unpack(b'ii', response.tobytes())
    return first if kind is int else first * (2.0 ** second)


def write_parameter(device, name: str, value) -> None:
    """Write one parameter with the wire format its type requires."""
    import usb.util
    if name in READ_ONLY:
        raise ValueError(f'{name} is read-only')
    parameter_id, offset, kind, _default, _doc = PARAMETERS[name]
    if kind is int:
        payload = struct.pack(b'iii', offset, int(value), 1)
    else:
        payload = struct.pack(b'ifi', offset, float(value), 0)
    device.ctrl_transfer(
        usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
        0, 0, parameter_id, payload, _CTRL_TIMEOUT)


def apply_profile(device, profile: dict) -> list:
    """Write a profile; return [(name, before, after, ok)] read back from the array."""
    results = []
    for name, value in profile.items():
        before = read_parameter(device, name)
        write_parameter(device, name, value)
        after = read_parameter(device, name)
        results.append((name, before, after, abs(float(after) - float(value)) < 1e-3))
    return results


def report(device) -> None:
    print(f'{"parameter":18s} {"value":>12s} {"default":>12s}   note')
    for name, (_id, _offset, kind, default, doc) in PARAMETERS.items():
        try:
            value = read_parameter(device, name)
        except Exception as exc:  # noqa: BLE001
            print(f'{name:18s} {"unreadable":>12s} {"":>12s}   {exc}')
            continue
        formatted = f'{value:.4g}' if kind is float else str(value)
        flag = '' if abs(float(value) - float(default)) < 1e-6 else '  <-- not default'
        print(f'{name:18s} {formatted:>12s} {default:>12} {flag}   {doc}')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--profile', choices=sorted(PROFILES), help='apply a tuned profile')
    group.add_argument('--restore-defaults', action='store_true',
                       help='write the published firmware defaults back')
    args = parser.parse_args(argv)

    try:
        device = find_device()
    except Exception as exc:  # noqa: BLE001
        print(f'FAIL: pyusb unavailable ({exc})')
        return 2
    if device is None:
        print('FAIL: no ReSpeaker Mic Array v2.0 (2886:0018) on this host.')
        return 2

    profile = None
    if args.profile:
        print(f'applying the {args.profile} profile (volatile: lost on power cycle)')
        profile = PROFILES[args.profile]
    elif args.restore_defaults:
        print('restoring published firmware defaults')
        profile = {name: spec[3] for name, spec in PARAMETERS.items() if name not in READ_ONLY}
    if profile is not None:
        for name, before, after, ok in apply_profile(device, profile):
            print(f'  {name:18s} {before:>10.4g} -> {after:<10.4g} {"ok" if ok else "MISMATCH"}')
        print()
    report(device)
    return 0


if __name__ == '__main__':
    sys.exit(main())
