"""The ReSpeaker DSP wire format, pinned without a microphone.

Two register mistakes are silent: the wrong offset sets a different parameter,
and the integer wire format applied to a float parameter writes nonsense that
reads back as a plausible number.
"""

import struct

import pytest

usb = pytest.importorskip('usb.util')

from come_here_audio import respeaker_tune as tune  # noqa: E402


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


def test_far_field_profile_turns_agc_on_and_lowers_vad_threshold():
    assert tune.FAR_FIELD_PROFILE['AGCONOFF'] == 1
    assert tune.FAR_FIELD_PROFILE['AGCMAXGAIN'] >= 1000.0
    assert tune.FAR_FIELD_PROFILE['GAMMAVAD_SR'] < tune.PARAMETERS['GAMMAVAD_SR'][3]
