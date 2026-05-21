"""Safety-behavior tests for BehaviorNode (require a ROS 2 runtime).

Covers the skeptic-review Priority 0 fixes:
  C1 — _stop_motion() must halt locomotion via cmd_velocity (the topic the
       bridge's gait republisher actually acts on), not the vestigial cmd_move.
  C3 — destroy_node() must emit a stop so the robot does not coast on a stale
       setpoint when the behavior node is killed.
  C4 — sensor callbacks must reject non-finite / out-of-range values before
       they become robot motion.
"""

import pytest
import rclpy
from std_msgs.msg import Float64MultiArray

from come_here_behavior.behavior_node import BehaviorNode


class FakePub:
    """Records published messages instead of sending them over DDS."""

    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


@pytest.fixture(scope='module', autouse=True)
def _rclpy_runtime():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = BehaviorNode()
    yield n
    n.destroy_node()


# -- C1: _stop_motion halts locomotion on the topic the bridge acts on --

def test_stop_motion_publishes_zero_on_cmd_velocity(node):
    vel = FakePub()
    node._velocity_pub = vel
    node._stop_motion()
    assert len(vel.msgs) == 1, 'expected exactly one cmd_velocity message'
    assert list(vel.msgs[0].data) == [0.0, 0.0]


def test_stop_motion_emits_multiarray(node):
    vel = FakePub()
    node._velocity_pub = vel
    node._stop_motion()
    assert isinstance(vel.msgs[0], Float64MultiArray)


# -- C3: destroy_node emits a stop so the robot does not coast on shutdown --

def test_destroy_node_publishes_stop():
    n = BehaviorNode()
    vel = FakePub()
    n._velocity_pub = vel
    n.destroy_node()
    assert any(list(m.data) == [0.0, 0.0] for m in vel.msgs), \
        'destroy_node must publish a zero cmd_velocity'


# -- C4: reject non-finite / out-of-range sensor inputs --

def test_direction_cb_rejects_nan_azimuth(node):
    node._last_azimuth = 0.5
    node._last_dir_confidence = 0.9
    msg = Float64MultiArray()
    msg.data = [float('nan'), 0.9]
    node._direction_cb(msg)
    assert node._last_azimuth == 0.5
    assert node._last_dir_confidence == 0.9


def test_direction_cb_rejects_out_of_range_azimuth(node):
    node._last_azimuth = 0.0
    msg = Float64MultiArray()
    msg.data = [50.0, 0.9]  # ~2864 deg, far outside [-pi, pi]
    node._direction_cb(msg)
    assert node._last_azimuth == 0.0


def test_direction_cb_accepts_valid_azimuth(node):
    msg = Float64MultiArray()
    msg.data = [0.6, 0.9]
    node._direction_cb(msg)
    assert node._last_azimuth == 0.6
    assert node._last_dir_confidence == 0.9


def test_person_cb_rejects_non_finite(node):
    node._person_detected = False
    msg = Float64MultiArray()
    msg.data = [float('inf'), 1.0, 0.9, 1.0]
    node._person_cb(msg)
    assert node._person_detected is False


def test_person_cb_accepts_valid_detection(node):
    node._person_detected = False
    msg = Float64MultiArray()
    msg.data = [0.1, 1.5, 0.9, 1.0]
    node._person_cb(msg)
    assert node._person_detected is True
    assert node._person_distance == 1.5
