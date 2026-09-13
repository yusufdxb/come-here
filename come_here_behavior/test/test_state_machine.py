"""State enum contract: /come_here/state publishes these names."""

from come_here_behavior.behavior_node import State


def test_all_states_exist():
    expected = {
        'IDLE',
        'LISTENING',
        'TURN_TO_SOUND',
        'ACQUIRE_PERSON',
        'ALIGN',
        'WALK',
        'ARRIVED',
        'SIT_AND_IDENTIFY',
    }
    assert {s.name for s in State} == expected


def test_align_and_walk_are_separate_states():
    # ALIGN (yaw only) and WALK (forward only) are distinct so the published
    # state shows which single-axis phase the robot is in.
    assert State.ALIGN != State.WALK
