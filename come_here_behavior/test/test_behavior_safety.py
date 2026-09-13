"""Node-level tests for BehaviorNode (need a ROS 2 runtime, not the Unitree SDK).

Scenario logic is tested without ROS in test_fsm.py. These tests check the
adapter wiring: messages in, published commands out.

  C1  stops go out on /come_here/cmd_velocity, the topic ALIGN and WALK drive
  C3  destroy_node() publishes a zero cmd_velocity (signal delivery at process
      level is covered by test_shutdown_stop.py)
  C4  malformed or non-finite perception never produces motion
"""

import json
import math

import pytest
import rclpy
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Float64MultiArray, String

from come_here_behavior.behavior_node import BehaviorNode

PUBLISHERS = (
    '_velocity_pub', '_rotate_pub', '_say_pub', '_sit_pub', '_stand_pub',
    '_face_req_pub', '_state_pub', '_trial_pub',
)


class FakePub:
    """Records published messages instead of sending them over DDS."""

    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture(scope='module', autouse=True)
def _rclpy_runtime():
    rclpy.init()
    yield
    rclpy.shutdown()


def _make_node(**params):
    params.setdefault('trial_log_enabled', False)
    node = BehaviorNode(
        parameter_overrides=[Parameter(k, value=v) for k, v in params.items()]
    )
    for name in PUBLISHERS:
        setattr(node, name, FakePub())
    node._now = FakeClock()
    return node


@pytest.fixture
def node():
    n = _make_node()
    yield n
    n.destroy_node()


def _person(node, *values):
    m = Float64MultiArray()
    m.data = [float(v) for v in values]
    node._person_cb(m)


def _wake(node, phrase='come here'):
    m = String()
    m.data = phrase
    node._wake_cb(m)


def _tick(node, n=1):
    for _ in range(n):
        node._tick()
        node._now.t += 0.1


def _velocities(node):
    return [list(m.data) for m in node._velocity_pub.msgs]


def _moving(node):
    return [v for v in _velocities(node) if v != [0.0, 0.0]]


CENTERED = (0.0, 2.0, 0.9, 1.0, 0.5, 2.0, 0.1)
CLOSE = (0.0, 1.8, 0.9, 1.0, 0.8, 2.0, 0.1)
MISS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1)


def _walk(node):
    _wake(node)
    for _ in range(2):
        _person(node, *CENTERED)
        _tick(node)


# -- C1 --

def test_stop_motion_publishes_zero_on_cmd_velocity(node):
    node._stop_motion()
    assert _velocities(node) == [[0.0, 0.0]]


def test_walk_drives_cmd_velocity_forward_only(node):
    _walk(node)
    assert _velocities(node)[-1] == [0.6, 0.0]
    assert node._state_pub.msgs[-1].data == 'WALK'


def test_lost_person_stop_goes_out_on_cmd_velocity(node):
    _walk(node)
    for _ in range(5):
        _person(node, *MISS)
        _tick(node)
    assert _velocities(node)[-1] == [0.0, 0.0]
    assert node._state_pub.msgs[-1].data == 'ACQUIRE_PERSON'


def test_estop_topic_stops_behavior(node):
    _walk(node)
    m = Bool()
    m.data = True
    node._estop_cb(m)
    assert _velocities(node)[-1] == [0.0, 0.0]
    _tick(node)
    assert node._state_pub.msgs[-1].data == 'IDLE'


# -- C3 --

def test_destroy_node_publishes_zero_cmd_velocity():
    n = _make_node()
    _walk(n)
    vel = n._velocity_pub
    n.destroy_node()
    assert list(vel.msgs[-1].data) == [0.0, 0.0]


# -- C4 --

@pytest.mark.parametrize('values', [
    (math.nan, 2.0, 0.9, 1.0),
    (0.0, math.inf, 0.9, 1.0, 0.5),
    (0.0, 2.0, 0.9),
    (0.0, 2.0, 0.9, 1.0, 0.5, 2.0, 0.1, 7.0),
    (4.0, 2.0, 0.9, 1.0, 0.5),
])
def test_bad_person_messages_never_move(node, values):
    _wake(node)
    for _ in range(20):
        _person(node, *values)
        _tick(node)
    assert _moving(node) == []


def test_non_finite_audio_direction_never_rotates():
    n = _make_node(skip_turn_to_sound=False)
    try:
        m = Float64MultiArray()
        m.data = [math.nan, 0.9]
        n._direction_cb(m)
        _wake(n)
        _tick(n, 5)
        assert n._rotate_pub.msgs == []
    finally:
        n.destroy_node()


# -- speech, config, trial evidence --

def test_wake_says_i_am_coming(node):
    _wake(node)
    assert [m.data for m in node._say_pub.msgs] == ['I am coming']


def test_invalid_config_refuses_to_start():
    with pytest.raises(ValueError):
        BehaviorNode(parameter_overrides=[
            Parameter('approach_align_threshold_rad', value=0.5),
            Parameter('trial_log_enabled', value=False),
        ])


def test_trial_record_is_published_and_written(tmp_path):
    n = _make_node(trial_log_enabled=True, trial_log_dir=str(tmp_path))
    try:
        detail = String()
        detail.data = json.dumps({'confidence': 0.82, 'speech_end_to_publish_s': 1.4})
        n._wake_detail_cb(detail)
        _walk(n)
        _person(n, *CLOSE)
        _tick(n, 25)  # arrive, then the 2 s arrival hold
        [msg] = n._trial_pub.msgs
        record = json.loads(msg.data)
        assert record['success'] is True
        assert record['stop_reason'] == 'arrived_bbox'
        assert record['wake']['source'] == 'audio'
        assert record['wake']['confidence'] == 0.82
        assert record['run_id'] and record['git_commit']
        assert record['config']['bbox_stop_fraction'] == 0.75
        lines = (tmp_path / 'trials.jsonl').read_text().splitlines()
        assert len(lines) == 1 and json.loads(lines[0])['run_id'] == record['run_id']
    finally:
        n.destroy_node()


def test_manually_injected_wake_is_marked_as_topic_source():
    n = _make_node()
    try:
        _wake(n)
        n._estop_cb(Bool(data=True))
        record = json.loads(n._trial_pub.msgs[-1].data)
        assert record['wake']['source'] == 'topic'
        assert record['stop_reason'] == 'estop'
    finally:
        n.destroy_node()
