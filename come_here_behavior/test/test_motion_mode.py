"""Motion-mode helpers: the read-only CheckMode path can never become Damp."""

import pytest

from come_here_behavior.motion_mode import (
    MOTION_SWITCHER_CHECK_MODE_API_ID,
    MOTION_SWITCHER_REQUEST_TOPIC,
    SPORT_API_DAMP,
    VALID_MOTION_MODES,
    check_sport_api_ids,
    mode_verdict,
    parse_mode_response,
)


def test_check_mode_goes_to_the_motion_switcher_topic_not_sport():
    assert MOTION_SWITCHER_REQUEST_TOPIC == '/api/motion_switcher/request'
    assert 'sport' not in MOTION_SWITCHER_REQUEST_TOPIC
    assert MOTION_SWITCHER_CHECK_MODE_API_ID == SPORT_API_DAMP == 1001


def test_normal_is_never_a_valid_mode():
    assert 'normal' not in VALID_MOTION_MODES
    assert set(VALID_MOTION_MODES) == {'mcf', 'ai'}


@pytest.mark.parametrize('data,name', [
    ('{"form":"0","name":"mcf"}', 'mcf'),
    ('{"form":"0","name":"ai"}', 'ai'),
    ('{"form":"0"}', None),
    ('not json', None),
    ('[1, 2]', None),
    ('', None),
])
def test_parse_mode_response(data, name):
    assert parse_mode_response(data) == name


def test_verdict_requires_the_exact_mode():
    assert mode_verdict('mcf', 'mcf')[0] is True
    assert mode_verdict('ai', 'mcf')[0] is False
    assert mode_verdict(None, 'mcf')[0] is False


def test_sport_api_1001_is_refused():
    with pytest.raises(ValueError):
        check_sport_api_ids(stop_move_api_id=1001)
    check_sport_api_ids(move_api_id=1008, stop_move_api_id=1003, sit_api_id=1009,
                        stand_api_id=1002)
