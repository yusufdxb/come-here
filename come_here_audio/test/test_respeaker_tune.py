"""The ReSpeaker DSP wire format, pinned without a microphone.

Two register mistakes are silent: the wrong offset sets a different parameter,
and the integer wire format applied to a float parameter writes nonsense that
reads back as a plausible number.
"""

import struct

import pytest

from come_here_audio import respeaker_tune as tune

try:
    import usb.util  # noqa: F401  (respeaker_tune imports it inside each call)
    _PYUSB_AVAILABLE = True
except ImportError:
    _PYUSB_AVAILABLE = False

# A module-level pytest.importorskip would mark the shared `test` package
# skipped and silently drop every sibling test file from the run.
pytestmark = pytest.mark.skipif(not _PYUSB_AVAILABLE, reason='pyusb not installed')


class FakeDevice:
    def __init__(self, response=b'\x01\x00\x00\x00\x00\x00\x00\x00'):
        self.calls = []
        self._response = response

    def ctrl_transfer(self, request_type, request, value, index, data, timeout):
        self.calls.append((request_type, request, value, index, data, timeout))
        if isinstance(data, int):  # a read: data is the length

            class Response:
                def __init__(self, raw):
                    self._raw = raw

                def tobytes(self):
                    return self._raw

            return Response(self._response)
        return len(data)


def test_published_offsets_are_used():
    assert tune.PARAMETERS['AGCONOFF'][:3] == (19, 0, int)
    assert tune.PARAMETERS['AGCMAXGAIN'][:3] == (19, 1, float)
    assert tune.PARAMETERS['AGCDESIREDLEVEL'][:3] == (19, 2, float)
    assert tune.PARAMETERS['AGCTIME'][:3] == (19, 4, float)
    assert tune.PARAMETERS['GAMMAVAD_SR'][:3] == (19, 39, float)
    assert tune.PARAMETERS['VOICEACTIVITY'][:3] == (19, 32, int)
    assert tune.PARAMETERS['DOAANGLE'][:3] == (21, 0, int)


def test_integer_parameter_uses_integer_wire_format():
    device = FakeDevice()
    tune.write_parameter(device, 'AGCONOFF', 1)
    assert device.calls[-1][4] == struct.pack(b'iii', 0, 1, 1)


def test_float_parameter_uses_float_wire_format():
    device = FakeDevice()
    tune.write_parameter(device, 'AGCMAXGAIN', 1000.0)
    assert device.calls[-1][4] == struct.pack(b'ifi', 1, 1000.0, 0)


def test_reads_set_the_command_bits_the_firmware_expects():
    device = FakeDevice()
    tune.read_parameter(device, 'AGCONOFF')
    assert device.calls[-1][2] == 0x80 | 0x40 | 0
    tune.read_parameter(device, 'GAMMAVAD_SR')
    assert device.calls[-1][2] == 0x80 | 39


def test_read_only_parameters_refuse_writes():
    device = FakeDevice()
    for name in tune.READ_ONLY:
        with pytest.raises(ValueError):
            tune.write_parameter(device, name, 1)


def test_far_field_profile_only_writes_writable_parameters():
    for name in tune.FAR_FIELD_PROFILE:
        assert name in tune.PARAMETERS
        assert name not in tune.READ_ONLY


def test_defaults_match_the_published_firmware_defaults():
    # respeaker/usb_4_mic_array tuning.py: AGCMAXGAIN defaults to 31.6 (30 dB).
    # GAMMAVAD_SR is documented in dB, '[-inf .. 60] dB (default: 3.5dB)', and the
    # SDK's set_vad_threshold writes the dB value directly.
    assert tune.PARAMETERS['AGCMAXGAIN'][3] == 31.6
    assert tune.PARAMETERS['GAMMAVAD_SR'][3] == 3.5


def test_far_field_profile_turns_agc_on_and_uncaps_it():
    assert tune.FAR_FIELD_PROFILE['AGCONOFF'] == 1
    assert tune.FAR_FIELD_PROFILE['AGCMAXGAIN'] > tune.PARAMETERS['AGCMAXGAIN'][3]


def test_far_field_profile_lowers_the_vad_threshold():
    # Lab 2026-09-14: the array read GAMMAVAD_SR 15.0 (our reader and the SDK's
    # tuning.py agree) and VOICEACTIVITY stayed 0 for normal speech at 2 m. ODIN's
    # far-field value 2.0 dB is below both that and the SDK default.
    assert tune.FAR_FIELD_PROFILE['GAMMAVAD_SR'] == 2.0
    assert tune.FAR_FIELD_PROFILE['GAMMAVAD_SR'] < tune.PARAMETERS['GAMMAVAD_SR'][3] < 15.0


# Published min/max from the tuning table, checked against every profile value.
OFFICIAL_RANGES = {
    'AGCONOFF': (0, 1), 'AGCMAXGAIN': (1, 1000), 'AGCDESIREDLEVEL': (1e-8, 0.99),
    'STATNOISEONOFF': (0, 1), 'GAMMA_NS': (0, 3), 'MIN_NS': (0, 1), 'HPFONOFF': (0, 3),
    'GAMMAVAD_SR': (0, 60), 'AGCGAIN': (1, 1000),
}


def test_agc_gain_is_writable_and_seeded_by_the_far_field_profile():
    # SDK tuning.py: 'AGCGAIN': (19, 3, 'float', 1000, 1, 'rw', ...). Lab 09-15:
    # a cold array (gain 1.25) never opened the gate for a 2.5 m talker.
    assert 'AGCGAIN' not in tune.READ_ONLY
    assert tune.PARAMETERS['AGCGAIN'][:3] == (19, 3, float)
    device = FakeDevice()
    tune.write_parameter(device, 'AGCGAIN', 10.0)
    assert device.calls[-1][4] == struct.pack(b'ifi', 3, 10.0, 0)
    assert tune.FAR_FIELD_PROFILE['AGCGAIN'] == 10.0


def test_far_field_profile_values_are_inside_the_published_ranges():
    for name, value in tune.FAR_FIELD_PROFILE.items():
        low, high = OFFICIAL_RANGES[name]
        assert low <= value <= high, name
