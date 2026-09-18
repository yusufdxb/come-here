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


def _software_doa_node():
    """A mock node whose calibration loader is exercised directly."""
    return AudioNode(parameter_overrides=[Parameter('use_mock', value=True)])


def test_calibration_file_replaces_offset_and_mirror(tmp_path):
    cal = tmp_path / 'doa_calibration.json'
    cal.write_text(json.dumps({'offset_deg': -31.5, 'mirror': True, 'measured_at': 'now'}))
    node = _software_doa_node()
    try:
        label = node._load_doa_calibration(str(cal), True, 'software', False)
        assert label == 'file (now)'
        assert node._doa_offset_deg == -31.5 and node._doa_mirror is True
    finally:
        node.destroy_node()


def test_missing_required_calibration_blocks_the_bearing(tmp_path):
    node = _software_doa_node()
    try:
        node._doa_offset_deg, node._doa_mirror = 12.0, False
        label = node._load_doa_calibration(str(tmp_path / 'absent.json'), True, 'software', False)
        assert label == 'MISSING'
        assert (node._doa_offset_deg, node._doa_mirror) == (12.0, False)
    finally:
        node.destroy_node()


def test_bad_calibration_values_are_not_applied(tmp_path):
    cal = tmp_path / 'doa_calibration.json'
    cal.write_text(json.dumps({'offset_deg': 'nan', 'mirror': 'yes'}))
    node = _software_doa_node()
    try:
        assert node._load_doa_calibration(str(cal), False, 'software', False) == 'parameters'
        assert node._load_doa_calibration(str(cal), True, 'software', False) == 'MISSING'
    finally:
        node.destroy_node()


def test_no_calibration_path_uses_parameters():
    node = _software_doa_node()
    try:
        assert node._load_doa_calibration('', True, 'software', False) == 'parameters'
        assert node._load_doa_calibration('/x.json', True, 'software', True) == 'n/a'
    finally:
        node.destroy_node()


def test_builtin_doa_publishes_the_utterance_bearing_before_the_wake():
    from come_here_audio.doa_association import DoaSelection
    node = AudioNode(parameter_overrides=[Parameter('use_mock', value=True)])
    calls = {}

    class FakeBuiltinDoa:
        def teardown(self):
            pass

        def get_direction_near(self, **kw):
            calls.update(kw)
            return DoaSelection(azimuth_rad=1.2, confidence=0.7, source='gate_window',
                                n_window=30, n_active=0, n_used=28, age_s=0.1, n_distinct=6)

    order = []

    class OrderedPub(FakePub):
        def __init__(self, label):
            super().__init__()
            self.label = label

        def publish(self, msg):
            order.append(self.label)
            super().publish(msg)

    try:
        node._direction_provider = FakeBuiltinDoa()
        node._dir_pub = OrderedPub('direction')
        node._detail_pub = OrderedPub('detail')
        node._wake_pub = OrderedPub('wake')

        class Detection:
            phrase, confidence, transcript, ratio, infer_ms = 'come here', 0.9, 'come here', 1.0, 5.0
            t_speech_end, t_speech_start, doa = 100.0, 98.8, None

        node._wake_detector.check = lambda: Detection()
        node._tick()
        assert order == ['direction', 'detail', 'wake']
        assert node._dir_pub.msgs[0].data.tolist() == [1.2, 0.7]
        assert calls['speech_end_s'] == 100.0 and calls['speech_start_s'] == 98.8
        detail = json.loads(node._detail_pub.msgs[0].data)
        assert detail['doa_source'] == 'gate_window' and detail['doa_n_used'] == 28
    finally:
        node.destroy_node()


def test_builtin_doa_without_samples_publishes_no_direction():
    node = AudioNode(parameter_overrides=[Parameter('use_mock', value=True)])

    class EmptyBuiltinDoa:
        def teardown(self):
            pass

        def get_direction_near(self, **kw):
            return None

    try:
        node._direction_provider = EmptyBuiltinDoa()
        node._dir_pub, node._detail_pub, node._wake_pub = FakePub(), FakePub(), FakePub()

        class Detection:
            phrase, confidence, transcript, ratio, infer_ms = 'come here', 0.9, 'come here', 1.0, 5.0
            t_speech_end, t_speech_start, doa = 100.0, 99.0, None

        node._wake_detector.check = lambda: Detection()
        node._tick()
        assert node._dir_pub.msgs == []
        assert len(node._wake_pub.msgs) == 1
        assert json.loads(node._detail_pub.msgs[0].data)['doa_source'] == 'none'
    finally:
        node.destroy_node()
