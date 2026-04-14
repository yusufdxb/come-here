"""Basic state enum tests. Full integration tests require ROS 2 runtime."""

from come_here_behavior.behavior_node import State


def test_all_states_exist():
    expected = {
        'IDLE',
        'LISTENING',
        'TURN_TO_SOUND',
        'SEARCH_FOR_PERSON',
        'APPROACH_PERSON',
        'SIT_AND_IDENTIFY',
    }
    actual = {s.name for s in State}
    assert actual == expected


def test_sit_and_identify_is_terminal_sequence_state():
    # This is a documentation test: SIT_AND_IDENTIFY should exist and be
    # distinct from IDLE / APPROACH_PERSON. It represents the linear
    # sit -> face-detect -> speak -> stand -> IDLE sequence.
    assert State.SIT_AND_IDENTIFY != State.IDLE
    assert State.SIT_AND_IDENTIFY != State.APPROACH_PERSON
