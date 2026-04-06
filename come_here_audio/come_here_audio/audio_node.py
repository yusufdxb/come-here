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
from come_here_audio.wake_phrase_detector import (
    MockWakePhraseDetector,
    WakePhraseDetector,
)

# TODO: Replace with real message imports after come_here_msgs is built
# from come_here_msgs.msg import AudioDirection, WakePhrase
from std_msgs.msg import Float64MultiArray, String


class AudioNode(Node):
    def __init__(self):
        super().__init__('audio_node')

        # Parameters
        self.declare_parameter('use_mock', True)
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('mock_azimuth_rad', 0.0)

        use_mock = self.get_parameter('use_mock').value
        rate_hz = self.get_parameter('publish_rate_hz').value
        mock_azimuth = self.get_parameter('mock_azimuth_rad').value

        # Provider selection
        # TODO: Add real provider selection when mic hardware is known
        if use_mock:
            self._direction_provider: AudioDirectionProvider = MockAudioProvider(
                fixed_azimuth_rad=mock_azimuth
            )
            self._wake_detector: WakePhraseDetector = MockWakePhraseDetector()
            self.get_logger().info('Using MOCK audio providers (no real hardware)')
        else:
            # TODO: Instantiate real hardware provider here
            raise NotImplementedError(
                'Real audio provider not yet implemented. '
                'Set use_mock:=true or implement a real AudioDirectionProvider.'
            )

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

        # Mock trigger subscriber
        if use_mock:
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
