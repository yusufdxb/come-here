"""Basic state enum tests. Full integration tests require ROS 2 runtime."""

from come_here_behavior.behavior_node import State


def test_all_states_exist():
    expected = {'IDLE', 'LISTENING', 'TURN_TO_SOUND', 'SEARCH_FOR_PERSON', 'APPROACH_PERSON', 'STOP'}
    actual = {s.name for s in State}
    assert actual == expected
