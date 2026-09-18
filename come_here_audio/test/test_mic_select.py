"""The microphone resolver: never silently fall back to a source that may be mute."""

import pytest

from come_here_audio.mic_select import (
    MicNotFound,
    describe_inputs,
    is_far_field,
    select_input_device,
)

RESPEAKER = {'name': 'ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:3,0)', 'max_input_channels': 6}
RESPEAKER_2CH = {'name': 'ReSpeaker 4 Mic Array (UAC1.0) alias', 'max_input_channels': 2}
ONBOARD = {'name': 'USB Audio: - (hw:1,0)', 'max_input_channels': 2}
HDMI_OUT = {'name': 'HDA NVidia: HDMI 0', 'max_input_channels': 0}
PULSE = {'name': 'pulse', 'max_input_channels': 32}
DEFAULT = {'name': 'default', 'max_input_channels': 32}


def test_describe_inputs_skips_output_only_devices():
    assert describe_inputs([HDMI_OUT, ONBOARD]) == [(1, 'USB Audio: - (hw:1,0)', 2)]


def test_far_field_needs_both_name_and_channel_count():
    assert is_far_field(RESPEAKER['name'], 6)
    assert not is_far_field('ReSpeaker passthrough', 2)
    assert not is_far_field('USB Audio: - (hw:1,0)', 2)


def test_prefers_the_far_field_array():
    index, name, channels, far = select_input_device([HDMI_OUT, ONBOARD, RESPEAKER, DEFAULT])
    assert (index, channels, far) == (2, 6, True)


def test_falls_back_to_a_real_device_not_the_pulse_default():
    index, name, _, far = select_input_device([DEFAULT, PULSE, ONBOARD])
    assert (index, name, far) == (2, 'USB Audio: - (hw:1,0)', False)


def test_requested_name_picks_the_full_array_over_an_alias():
    index, _, channels, far = select_input_device([RESPEAKER_2CH, RESPEAKER], requested='respeaker')
    assert (index, channels, far) == (1, 6, True)


def test_explicit_index_wins():
    assert select_input_device([RESPEAKER, ONBOARD], requested='1')[0] == 1


def test_absent_requested_mic_raises_and_names_what_is_attached():
    with pytest.raises(MicNotFound) as err:
        select_input_device([ONBOARD], requested='respeaker')
    assert 'USB Audio' in str(err.value)


def test_absent_requested_index_raises():
    with pytest.raises(MicNotFound):
        select_input_device([ONBOARD], requested='9')


def test_no_capture_device_reports_none():
    assert select_input_device([HDMI_OUT])[0] is None
