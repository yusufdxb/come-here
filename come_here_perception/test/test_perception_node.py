"""Freshness tests for PerceptionNode (need a ROS 2 runtime, not ultralytics).

A dead camera must turn into explicit not-detected messages, and YOLO must run
once per new frame so one image cannot count as several detections.
"""

import pytest
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image

from come_here_perception.perception_node import PerceptionNode
from come_here_perception.person_detector import PersonDetector, PersonEstimate


class FakeDetector(PersonDetector):
    def __init__(self):
        self.frames = []
        self.detect_calls = 0
        self.estimate = PersonEstimate(
            bearing_rad=0.4, distance_m=2.0, confidence=0.9, detected=True,
            bbox_h_frac=0.6,
        )

    def setup(self):
        pass

    def update_frame(self, frame):
        self.frames.append(frame)

    def detect(self):
        self.detect_calls += 1
        return self.estimate

    def teardown(self):
        pass


class FakePub:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(list(msg.data))


class FakeClock:
    def __init__(self, t=500.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture(scope='module', autouse=True)
def _rclpy_runtime():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def setup():
    detector = FakeDetector()
    node = PerceptionNode(
        detector=detector,
        parameter_overrides=[Parameter('use_lidar_distance', value=False)],
    )
    node._pub = FakePub()
    node._now = FakeClock()
    yield node, detector
    node.destroy_node()


def _frame(width=4, height=2, size=None):
    msg = Image()
    msg.width, msg.height, msg.encoding = width, height, 'bgr8'
    msg.data = bytes(size if size is not None else width * height * 3)
    return msg


def test_no_frame_publishes_not_detected_without_running_yolo(setup):
    node, detector = setup
    node._tick()
    assert detector.detect_calls == 0
    [msg] = node._pub.msgs
    assert msg[3] == 0.0 and len(msg) == 7


def test_new_frame_is_detected_once(setup):
    node, detector = setup
    node._on_image(_frame())
    node._now.t += 0.05
    node._tick()
    node._tick()  # same frame again: must not re-detect or re-publish
    assert detector.detect_calls == 1
    [msg] = node._pub.msgs
    bearing, distance, conf, detected, bbox, source, age = msg
    assert (detected, conf, bbox, source) == (1.0, 0.9, 0.6, 1.0)
    assert age == pytest.approx(0.05)


def test_stale_camera_publishes_not_detected(setup):
    node, detector = setup
    node._on_image(_frame())
    node._tick()
    node._now.t += 1.5  # max_frame_age_s defaults to 1.0
    node._tick()
    assert detector.detect_calls == 1
    assert node._pub.msgs[-1][3] == 0.0


def test_bearing_is_published_raw(setup):
    node, detector = setup
    node._on_image(_frame())
    node._tick()
    detector.estimate = PersonEstimate(0.0, 2.0, 0.9, True, 0.6)
    node._on_image(_frame())
    node._tick()
    assert [m[0] for m in node._pub.msgs] == [0.4, 0.0]


def test_malformed_frame_is_dropped(setup):
    node, detector = setup
    node._on_image(_frame(size=5))
    assert detector.frames == []
    node._tick()
    assert detector.detect_calls == 0
    assert node._pub.msgs[-1][3] == 0.0
