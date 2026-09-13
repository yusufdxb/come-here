"""ROS 2 node for audio perception: wake phrase detection and optional DOA.

Publishes:
  /come_here/wake_phrase      (std_msgs/String) canonical phrase, e.g. "come here"
  /come_here/wake_detail      (std_msgs/String) JSON for the trial log: confidence,
                              transcript, speech_end_to_publish_s, infer_ms, gate
  /come_here/audio_health     (std_msgs/String) JSON every health_period_s: mic,
                              capture age, rms, noise floor, gate
  /come_here/audio_direction  (std_msgs/Float64MultiArray) [azimuth, confidence],
                              only when enable_doa is true

Subscribes:
  /come_here/mock_trigger     (std_msgs/Bool) mock wake phrase trigger

use_mock forces the mock wake detector and mock DOA, so mock launches never
need pyusb, sounddevice or faster-whisper.
"""

import json
import time

from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray, String

from come_here_audio.audio_direction_provider import AudioDirectionProvider
from come_here_audio.mock_audio_provider import MockAudioProvider
from come_here_audio.wake_phrase_detector import (
    MockWakePhraseDetector,
    WakePhraseDetector,
)

# Hardware-backed providers are imported lazily inside the non-mock branches so
# mock-mode launches do not require pyusb / faster-whisper to be installed.


class AudioNode(Node):
    def __init__(self, **kwargs):
        super().__init__('audio_node', **kwargs)

        p = self.declare_parameter
        p('use_mock', True)
        p('wake_detector', 'mock')  # 'mock' or 'whisper'
        p('publish_rate_hz', 10.0)
        p('health_period_s', 5.0)
        p('enable_doa', True)
        p('mock_azimuth_rad', 0.0)
        p('respeaker_frame_offset_deg', 0.0)
        p('whisper_model_size', 'base.en')
        p('whisper_device', 'cpu')
        p('whisper_compute_type', 'int8')
        p('whisper_cpu_threads', 2)
        p('whisper_adapter_path', '')  # path to LoRA adapter
        p('whisper_confidence_threshold', 0.4)
        p('whisper_no_speech_threshold', 0.5)
        p('whisper_vad_filter', True)
        p('phrase_ratio_threshold', 0.80)
        p('wake_cooldown_s', 3.0)
        # Microphone: resolved by name; the far-field array is preferred.
        p('mic_device', 'ReSpeaker')
        p('mic_prefer_far_field', True)
        # ReSpeaker ch 0 is the DSP output (beamformer + AEC + AGC + NS). Open
        # the device mono and PortAudio hands us that channel.
        p('mic_channels', 1)
        p('mic_beam_channel', 0)
        p('mic_gain', 1.0)
        # Far-field front end (see whisper_phrase_detector).
        p('respeaker_profile', 'none')  # 'none' or 'far_field'
        p('adaptive_gate', True)
        p('gate_snr_margin', 2.0)
        p('gate_floor_rms', 0.003)
        p('calibration_s', 1.0)
        p('preroll_sec', 0.25)
        p('utterance_rms_threshold', 0.015)
        p('segment_peak_threshold', 0.02)
        p('silence_to_end_sec', 0.70)
        p('min_utterance_sec', 0.20)
        p('max_utterance_sec', 4.50)

        g = lambda name: self.get_parameter(name).value  # noqa: E731
        use_mock = bool(g('use_mock'))
        rate_hz = float(g('publish_rate_hz'))
        self._mic_label = 'mock'

        # Direction provider
        self._direction_provider: AudioDirectionProvider | None = None
        if use_mock:
            self._direction_provider = MockAudioProvider(fixed_azimuth_rad=g('mock_azimuth_rad'))
            self.get_logger().info('Using MOCK audio direction provider')
        elif g('enable_doa'):
            from come_here_audio.respeaker_doa_provider import ReSpeakerDOAProvider
            self._direction_provider = ReSpeakerDOAProvider(
                frame_offset_deg=g('respeaker_frame_offset_deg')
            )
            self.get_logger().info('Using RESPEAKER audio direction provider')
        else:
            self.get_logger().info('DOA disabled (enable_doa=false)')

        # Wake phrase detector
        if not use_mock and g('wake_detector') == 'whisper':
            self._apply_respeaker_profile(str(g('respeaker_profile')))
            mic_index = self._resolve_mic(str(g('mic_device')), bool(g('mic_prefer_far_field')),
                                          int(g('mic_channels')))
            from come_here_audio.whisper_phrase_detector import WhisperPhraseDetector
            adapter_path = g('whisper_adapter_path') or None
            self._wake_detector: WakePhraseDetector = WhisperPhraseDetector(
                model_size=g('whisper_model_size'),
                device=g('whisper_device'),
                compute_type=g('whisper_compute_type'),
                cpu_threads=int(g('whisper_cpu_threads')),
                adapter_path=adapter_path,
                mic_device=mic_index,
                mic_channels=int(g('mic_channels')),
                mic_beam_channel=int(g('mic_beam_channel')),
                mic_gain=float(g('mic_gain')),
                confidence_threshold=float(g('whisper_confidence_threshold')),
                no_speech_threshold=float(g('whisper_no_speech_threshold')),
                phrase_ratio_threshold=float(g('phrase_ratio_threshold')),
                whisper_vad_filter=bool(g('whisper_vad_filter')),
                cooldown_s=float(g('wake_cooldown_s')),
                adaptive_gate=bool(g('adaptive_gate')),
                gate_snr_margin=float(g('gate_snr_margin')),
                gate_floor_rms=float(g('gate_floor_rms')),
                calibration_s=float(g('calibration_s')),
                preroll_sec=float(g('preroll_sec')),
                utterance_rms_threshold=float(g('utterance_rms_threshold')),
                segment_peak_threshold=float(g('segment_peak_threshold')),
                silence_to_end_sec=float(g('silence_to_end_sec')),
                min_utterance_sec=float(g('min_utterance_sec')),
                max_utterance_sec=float(g('max_utterance_sec')),
            )
            self._detector_name = 'whisper'
            self.get_logger().info(
                f'Using WHISPER wake detector ({g("whisper_model_size")}, '
                f'cpu_threads={g("whisper_cpu_threads")}, adaptive_gate={g("adaptive_gate")})'
            )
        else:
            self._wake_detector = MockWakePhraseDetector()
            self._detector_name = 'mock'
            self.get_logger().info('Using MOCK wake phrase detector')

        if self._direction_provider is not None:
            self._direction_provider.setup()
        self._wake_detector.setup()

        self._dir_pub = self.create_publisher(Float64MultiArray, '/come_here/audio_direction', 10)
        self._wake_pub = self.create_publisher(String, '/come_here/wake_phrase', 10)
        self._detail_pub = self.create_publisher(String, '/come_here/wake_detail', 10)
        self._health_pub = self.create_publisher(String, '/come_here/audio_health', 10)

        if isinstance(self._wake_detector, MockWakePhraseDetector):
            self._mock_sub = self.create_subscription(
                Bool, '/come_here/mock_trigger', self._mock_trigger_cb, 10
            )

        self._timer = self.create_timer(1.0 / rate_hz, self._tick)
        self._health_timer = self.create_timer(float(g('health_period_s')), self._health_tick)
        self.get_logger().info(f'Audio node started at {rate_hz} Hz')

    def _apply_respeaker_profile(self, profile: str) -> None:
        """Push a DSP profile to the array. Never fatal: a mic that records beats
        a node that does not start, and the profile is an optimisation."""
        if profile in ('', 'none'):
            self.get_logger().info('respeaker_profile=none: array keeps its power-up registers')
            return
        try:
            from come_here_audio import respeaker_tune
            chosen = respeaker_tune.PROFILES[profile]
            device = respeaker_tune.find_device()
            if device is None:
                self.get_logger().warn('No ReSpeaker on USB to tune')
                return
            for name, before, after, ok in respeaker_tune.apply_profile(device, chosen):
                self.get_logger().info(
                    f'respeaker {name}: {before:.4g} -> {after:.4g} {"ok" if ok else "MISMATCH"}'
                )
            self.get_logger().warn(
                f'ReSpeaker {profile} DSP profile applied (volatile: reset on power cycle)'
            )
        except Exception as exc:  # noqa: BLE001 - optimisation, not a gate
            self.get_logger().warn(f'Could not apply ReSpeaker profile {profile!r}: {exc}')

    def _resolve_mic(self, requested: str, prefer_far_field: bool, channels_needed: int):
        from come_here_audio.mic_select import resolve
        index, name, channels, far_field = resolve(requested, prefer_far_field)
        if index is None:
            raise RuntimeError('No audio capture device found')
        if channels < channels_needed:
            raise RuntimeError(f'{name} has {channels} input channels, need {channels_needed}')
        self._mic_label = f'{index}:{name} ({channels}ch, far_field={far_field})'
        if far_field:
            self.get_logger().info(f'Microphone: {self._mic_label}')
        else:
            self.get_logger().warn(f'Microphone is NOT a far-field array: {self._mic_label}')
        return index

    def _mock_trigger_cb(self, msg: Bool):
        if isinstance(self._wake_detector, MockWakePhraseDetector):
            self._wake_detector.set_triggered(msg.data)
            self.get_logger().info('Mock wake phrase triggered')

    def _tick(self):
        if self._direction_provider is not None:
            estimate = self._direction_provider.get_direction()
            if estimate is not None:
                msg = Float64MultiArray()
                msg.data = [estimate.azimuth_rad, estimate.confidence]
                self._dir_pub.publish(msg)

        detection = self._wake_detector.check()
        if detection is None:
            return
        now = time.monotonic()
        detail = {
            'detector': self._detector_name,
            'phrase': detection.phrase,
            'confidence': round(float(detection.confidence), 3),
            'transcript': detection.transcript,
            'ratio': round(float(detection.ratio), 3),
            'infer_ms': None if detection.infer_ms is None else round(detection.infer_ms, 1),
            'speech_end_to_publish_s': (
                None if detection.t_speech_end is None
                else round(now - detection.t_speech_end, 3)
            ),
            'mic': self._mic_label,
        }
        if hasattr(self._wake_detector, 'capture_health'):
            health = self._wake_detector.capture_health()
            detail.update({k: health[k] for k in ('noise_floor', 'rms_gate')})
        detail_msg = String()
        detail_msg.data = json.dumps(detail)
        self._detail_pub.publish(detail_msg)
        wake = String()
        wake.data = detection.phrase
        self._wake_pub.publish(wake)
        self.get_logger().info(
            f'Wake phrase detected: "{detection.phrase}" (confidence={detection.confidence:.2f}, '
            f'heard="{detection.transcript}", latency={detail["speech_end_to_publish_s"]}s)'
        )

    def _health_tick(self):
        health = {'detector': self._detector_name, 'mic': self._mic_label}
        if hasattr(self._wake_detector, 'capture_health'):
            health.update(self._wake_detector.capture_health())
        msg = String()
        msg.data = json.dumps(health)
        self._health_pub.publish(msg)
        if health.get('capture_age_s', 0.0) > 2.0:
            self.get_logger().error(f'Microphone capture stalled: {health}')
        elif self._detector_name == 'whisper':
            self.get_logger().info(
                f'audio: rms={health["rms"]} noise_floor={health["noise_floor"]} '
                f'gate={health["rms_gate"]} calibrated={health["calibrated"]}'
            )

    def destroy_node(self):
        if self._direction_provider is not None:
            self._direction_provider.teardown()
        self._wake_detector.teardown()
        super().destroy_node()


def main(args=None):
    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = AudioNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # A second SIGINT can land during teardown or during interpreter
        # shutdown (e.g. threading._shutdown). Ignore it for the rest of the
        # process so shutdown stays quiet.
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
