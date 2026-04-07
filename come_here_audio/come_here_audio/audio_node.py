"""ROS 2 node for audio perception: direction estimation and wake phrase detection.

Publishes:
  /come_here/audio_direction  (come_here_msgs/AudioDirection)
  /come_here/wake_phrase      (come_here_msgs/WakePhrase)

Subscribes:
  /come_here/mock_trigger     (std_msgs/Bool) - mock wake phrase trigger
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Header

from come_here_audio.audio_direction_provider import AudioDirectionProvider
from come_here_audio.mock_audio_provider import MockAudioProvider
from come_here_audio.respeaker_doa_provider import ReSpeakerDOAProvider
from come_here_audio.wake_phrase_detector import (
    MockWakePhraseDetector,
    WakePhraseDetector,
)
from come_here_audio.whisper_phrase_detector import WhisperPhraseDetector

# TODO: Replace with real message imports after come_here_msgs is built
# from come_here_msgs.msg import AudioDirection, WakePhrase
from std_msgs.msg import Float64MultiArray, String


class AudioNode(Node):
    def __init__(self):
        super().__init__('audio_node')

        # Parameters
        self.declare_parameter('use_mock', True)
        self.declare_parameter('wake_detector', 'mock')  # 'mock' or 'whisper'
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('mock_azimuth_rad', 0.0)
        self.declare_parameter('whisper_model_size', 'base.en')
        self.declare_parameter('whisper_device', 'cpu')
        self.declare_parameter('whisper_chunk_duration_s', 2.0)
        self.declare_parameter('whisper_adapter_path', '')  # path to LoRA adapter

        use_mock = self.get_parameter('use_mock').value
        wake_detector_type = self.get_parameter('wake_detector').value
        rate_hz = self.get_parameter('publish_rate_hz').value
        mock_azimuth = self.get_parameter('mock_azimuth_rad').value

        # Direction provider selection
        self.declare_parameter('respeaker_frame_offset_deg', 0.0)
        frame_offset = self.get_parameter('respeaker_frame_offset_deg').value

        if use_mock:
            self._direction_provider: AudioDirectionProvider = MockAudioProvider(
                fixed_azimuth_rad=mock_azimuth
            )
            self.get_logger().info('Using MOCK audio direction provider')
        else:
            self._direction_provider = ReSpeakerDOAProvider(
                frame_offset_deg=frame_offset
            )
            self.get_logger().info('Using RESPEAKER audio direction provider')

        # Wake phrase detector selection
        self.declare_parameter('mic_device', 'hw:0,0')
        self.declare_parameter('mic_channels', 6)
        self.declare_parameter('mic_beam_channel', 1)

        if wake_detector_type == 'whisper':
            adapter_path = self.get_parameter('whisper_adapter_path').value
            adapter_path = adapter_path if adapter_path else None
            self._wake_detector: WakePhraseDetector = WhisperPhraseDetector(
                model_size=self.get_parameter('whisper_model_size').value,
                device=self.get_parameter('whisper_device').value,
                chunk_duration_s=self.get_parameter('whisper_chunk_duration_s').value,
                adapter_path=adapter_path,
                mic_device=self.get_parameter('mic_device').value,
                mic_channels=self.get_parameter('mic_channels').value,
                mic_beam_channel=self.get_parameter('mic_beam_channel').value,
            )
            label = 'WHISPER (fine-tuned)' if adapter_path else 'WHISPER (base)'
            self.get_logger().info(f'Using {label} wake phrase detector')
        else:
            self._wake_detector = MockWakePhraseDetector()
            self.get_logger().info('Using MOCK wake phrase detector')

        self._direction_provider.setup()
        self._wake_detector.setup()

        # Publishers - using std_msgs until come_here_msgs is built
        # TODO: Switch to come_here_msgs/AudioDirection and come_here_msgs/WakePhrase
        self._dir_pub = self.create_publisher(
            Float64MultiArray, '/come_here/audio_direction', 10
        )
        self._wake_pub = self.create_publisher(
            String, '/come_here/wake_phrase', 10
        )

        # Mock trigger subscriber (only when using mock wake detector)
        if isinstance(self._wake_detector, MockWakePhraseDetector):
            self._mock_sub = self.create_subscription(
                Bool, '/come_here/mock_trigger', self._mock_trigger_cb, 10
            )

        # Timer
        period = 1.0 / rate_hz
        self._timer = self.create_timer(period, self._tick)
        self.get_logger().info(f'Audio node started at {rate_hz} Hz')

    def _mock_trigger_cb(self, msg: Bool):
        if isinstance(self._wake_detector, MockWakePhraseDetector):
            self._wake_detector.set_triggered(msg.data)
            self.get_logger().info('Mock wake phrase triggered')

    def _tick(self):
        # Direction estimation
        estimate = self._direction_provider.get_direction()
        if estimate is not None:
            msg = Float64MultiArray()
            msg.data = [estimate.azimuth_rad, estimate.confidence]
            self._dir_pub.publish(msg)

        # Wake phrase check
        detection = self._wake_detector.check()
        if detection is not None:
            msg = String()
            msg.data = detection.phrase
            self._wake_pub.publish(msg)
            self.get_logger().info(
                f'Wake phrase detected: "{detection.phrase}" '
                f'(confidence={detection.confidence:.2f})'
            )

    def destroy_node(self):
        self._direction_provider.teardown()
        self._wake_detector.teardown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AudioNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
