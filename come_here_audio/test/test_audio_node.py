"""AudioNode wiring in mock mode (needs a ROS 2 runtime, no audio hardware)."""

import json

import pytest
import rclpy
from rclpy.parameter import Parameter
from std_msgs.msg import Bool

from come_here_audio.audio_node import AudioNode
from come_here_audio.wake_phrase_detector import MockWakePhraseDetector


class FakePub:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


@pytest.fixture(scope='module', autouse=True)
def _rclpy_runtime():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_use_mock_forces_the_mock_wake_detector():
    node = AudioNode(parameter_overrides=[
        Parameter('use_mock', value=True),
        Parameter('wake_detector', value='whisper'),
    ])
    try:
        assert isinstance(node._wake_detector, MockWakePhraseDetector)
    finally:
        node.destroy_node()


def test_wake_publishes_detail_then_phrase():
    node = AudioNode(parameter_overrides=[Parameter('use_mock', value=True)])
    order = []

    class OrderedPub(FakePub):
        def __init__(self, label):
            super().__init__()
            self.label = label

        def publish(self, msg):
            order.append(self.label)
            super().publish(msg)

    try:
        node._detail_pub = OrderedPub('detail')
        node._wake_pub = OrderedPub('wake')
        node._dir_pub = FakePub()
        node._mock_trigger_cb(Bool(data=True))
        node._tick()
        assert order == ['detail', 'wake']
        assert node._wake_pub.msgs[0].data == 'come here'
        detail = json.loads(node._detail_pub.msgs[0].data)
        assert detail['detector'] == 'mock' and detail['confidence'] == 0.95
    finally:
        node.destroy_node()
