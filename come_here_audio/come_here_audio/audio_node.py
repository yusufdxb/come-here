"""ROS 2 node for audio perception: wake phrase detection and optional DOA.

Publishes:
  /come_here/wake_phrase      (std_msgs/String) canonical phrase, e.g. "come here"
  /come_here/wake_detail      (std_msgs/String) JSON for the trial log: confidence,
                              transcript, speech_end_to_publish_s, infer_ms, gate
  /come_here/audio_health     (std_msgs/String) JSON every health_period_s: mic,
                              capture age, rms, noise floor, gate
  /come_here/audio_direction  (std_msgs/Float64MultiArray) [azimuth, confidence].
                              doa_source software: one message per wake, the
                              bearing of the utterance Whisper matched, from the
                              raw ReSpeaker capsules (srp_doa). doa_source
                              firmware: the array's DOAANGLE register, polled.

Subscribes:
  /come_here/mock_trigger     (std_msgs/Bool) mock wake phrase trigger

use_mock forces the mock wake detector and mock DOA, so mock launches never
need pyusb, sounddevice or faster-whisper.
"""

import json
import math
import os
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
        p('enable_doa', True)             # legacy: firmware register polling
        p('doa_source', 'none')           # 'software' | 'firmware' | 'none'
        p('doa_mirror', False)            # array mounted capsules-down
        p('doa_grid_deg', 2.0)
        p('doa_peak_full_scale', 0.25)    # SRP peak that counts as full confidence
        p('doa_firmware_advisory', True)  # also log DOAANGLE next to the software bearing
        p('mock_azimuth_rad', 0.0)
        p('respeaker_frame_offset_deg', 0.0)
        # doa_source firmware: DOAANGLE polled continuously, one bearing per wake
        # chosen from the samples inside that utterance (ODIN doa_association).
        p('doa_poll_rate_hz', 20.0)
        p('doa_pre_speech_s', 1.0)
        p('doa_post_speech_s', 0.3)
        p('doa_min_active_samples', 3)
        # true: a bearing the register already held before the utterance (never
        # moved while the caller spoke) is capped below the turn threshold. Off:
        # the flag is only logged (a caller where the last sound came from also
        # leaves the register unmoved).
        p('doa_reject_held_register', False)
        # scripts/calibrate_doa.py writes this file; when it loads, it replaces
        # respeaker_frame_offset_deg / doa_mirror. '' = use those two parameters.
        p('doa_calibration_path', '')
        # true: without a loaded calibration file no bearing is published, so the
        # robot never turns on an uncentered array (loud ERROR, not a silent bypass).
        p('require_doa_calibration', False)
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
        doa_source = str(g('doa_source')).lower()
        if doa_source not in ('software', 'firmware', 'none'):
            raise ValueError(f"doa_source must be software, firmware or none, got {doa_source!r}")
        if doa_source == 'none' and bool(g('enable_doa')):
            doa_source = 'firmware'
        self._doa_source = doa_source
        self._doa_offset_deg = float(g('respeaker_frame_offset_deg'))
        self._doa_mirror = bool(g('doa_mirror'))
        self._doa_calibration = self._load_doa_calibration(
            str(g('doa_calibration_path')), bool(g('require_doa_calibration')),
            doa_source, use_mock)
        self._doa_firmware_dev = None
        self._direction_provider: AudioDirectionProvider | None = None
        if use_mock:
            self._direction_provider = MockAudioProvider(fixed_azimuth_rad=g('mock_azimuth_rad'))
            self.get_logger().info('Using MOCK audio direction provider')
        elif doa_source == 'firmware':
            from come_here_audio.respeaker_doa_provider import ReSpeakerDOAProvider
            self._direction_provider = ReSpeakerDOAProvider(
                frame_offset_deg=g('respeaker_frame_offset_deg')
            )
            self.get_logger().info('Using RESPEAKER audio direction provider')
        elif doa_source == 'software':
            self.get_logger().info('Software DOA (SRP-PHAT on the raw capsules) per wake utterance')
        else:
            self.get_logger().info('DOA disabled (doa_source=none)')

        # Wake phrase detector
        if not use_mock and g('wake_detector') == 'whisper':
            self._apply_respeaker_profile(str(g('respeaker_profile')))
            mic_channels = int(g('mic_channels'))
            doa_estimator = None
            doa_channels = ()
            if doa_source == 'software':
                from come_here_audio.srp_doa import RESPEAKER_V2_RAW_CHANNELS, SrpPhatDoa
                doa_channels = RESPEAKER_V2_RAW_CHANNELS
                needed = max(doa_channels) + 1
                if mic_channels < needed:
                    self.get_logger().info(
                        f'software DOA needs the raw capsules: capturing {needed} channels '
                        f'(mic_channels was {mic_channels})')
                    mic_channels = needed
                doa_estimator = SrpPhatDoa(
                    grid_deg=float(g('doa_grid_deg')),
                    peak_full_scale=float(g('doa_peak_full_scale')),
                )
                if bool(g('doa_firmware_advisory')):
                    try:
                        from come_here_audio import respeaker_tune
                        self._doa_firmware_dev = respeaker_tune.find_device()
                    except Exception as exc:  # noqa: BLE001 - advisory only
                        self.get_logger().warn(f'firmware DOAANGLE advisory unavailable: {exc}')
            try:
                mic_index = self._resolve_mic(str(g('mic_device')),
                                              bool(g('mic_prefer_far_field')), mic_channels)
            except RuntimeError as exc:
                if doa_estimator is None or 'input channels' not in str(exc):
                    raise
                # 1-channel firmware on the array: keep the wake path alive, lose DOA.
                self.get_logger().error(
                    f'SOFTWARE DOA DISABLED, the array is not streaming its raw capsules: {exc}. '
                    'Flash the 6-channel ReSpeaker firmware to get turn-to-sound back.')
                doa_estimator = None
                doa_channels = ()
                mic_channels = int(g('mic_channels'))
                mic_index = self._resolve_mic(str(g('mic_device')),
                                              bool(g('mic_prefer_far_field')), mic_channels)
            from come_here_audio.whisper_phrase_detector import WhisperPhraseDetector
            adapter_path = g('whisper_adapter_path') or None
            self._wake_detector: WakePhraseDetector = WhisperPhraseDetector(
                model_size=g('whisper_model_size'),
                device=g('whisper_device'),
                compute_type=g('whisper_compute_type'),
                cpu_threads=int(g('whisper_cpu_threads')),
                adapter_path=adapter_path,
                mic_device=mic_index,
                mic_channels=mic_channels,
                mic_beam_channel=int(g('mic_beam_channel')),
                mic_gain=float(g('mic_gain')),
                doa_estimator=doa_estimator,
                doa_channels=doa_channels,
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
                f'cpu_threads={g("whisper_cpu_threads")}, adaptive_gate={g("adaptive_gate")}, '
                f'software_doa={doa_estimator is not None} '
                f'offset={self._doa_offset_deg:+.1f} deg mirror={self._doa_mirror})'
            )
        else:
            self._wake_detector = MockWakePhraseDetector()
            self._detector_name = 'mock'
            self.get_logger().info('Using MOCK wake phrase detector')

        self._doa_window = {
            'pre_s': float(g('doa_pre_speech_s')),
            'post_s': float(g('doa_post_speech_s')),
            'min_active_samples': int(g('doa_min_active_samples')),
            'reject_held': bool(g('doa_reject_held_register')),
        }
        if self._direction_provider is not None:
            self._direction_provider.setup()
            if hasattr(self._direction_provider, 'get_direction_near'):
                self._direction_provider.start_continuous(float(g('doa_poll_rate_hz')))
                self.get_logger().info(
                    f'Built-in DOA polled at {float(g("doa_poll_rate_hz")):.0f} Hz; one bearing '
                    f'per wake from the utterance window, offset {self._doa_offset_deg:+.1f} deg')
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

    def _load_doa_calibration(self, path: str, required: bool, doa_source: str,
                              use_mock: bool) -> str:
        """Apply scripts/calibrate_doa.py output. Returns a label for logs and health.

        'file (<measured_at>)': offset and mirror came from the file.
        'parameters': no file requested; yaml / launch offset and mirror are used.
        'MISSING': a file was required and did not load; no bearing is published.
        """
        if use_mock or doa_source != 'software':
            return 'n/a'
        if not path:
            self.get_logger().info(
                f'DOA calibration from parameters: offset {self._doa_offset_deg:+.1f} deg '
                f'mirror {self._doa_mirror}')
            return 'parameters'
        full = os.path.expanduser(path)
        try:
            with open(full) as f:
                cal = json.load(f)
            offset = float(cal['offset_deg'])
            mirror = cal['mirror']
            if not math.isfinite(offset) or not isinstance(mirror, bool):
                raise ValueError(f'bad values offset_deg={offset!r} mirror={mirror!r}')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reason = f'{type(exc).__name__}: {exc}'
        else:
            self._doa_offset_deg, self._doa_mirror = offset, mirror
            measured = cal.get('measured_at', '?')
            self.get_logger().info(
                f'DOA calibration loaded from {full}: offset {offset:+.1f} deg mirror {mirror} '
                f'(measured {measured}, ahead std {cal.get("ahead_circ_std_deg", "?")} deg)')
            return f'file ({measured})'
        if required:
            self.get_logger().error(
                f'DOA NOT CALIBRATED ({full}: {reason}). No bearing will be published, so the '
                'robot will NOT turn toward the voice. Run: python3 scripts/calibrate_doa.py')
            return 'MISSING'
        self.get_logger().warn(
            f'DOA calibration file not loaded ({full}: {reason}); using parameters: '
            f'offset {self._doa_offset_deg:+.1f} deg mirror {self._doa_mirror}')
        return 'parameters'

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

    def _utterance_direction(self, detection, detail):
        """Built-in DOA for this utterance: publish it before the wake. Returns a log note."""
        t_end = getattr(detection, 't_speech_end', None)
        if t_end is None:
            return ', direction unavailable (no speech timing)'
        try:
            sel = self._direction_provider.get_direction_near(
                speech_end_s=t_end, speech_start_s=getattr(detection, 't_speech_start', None),
                **self._doa_window)
        except Exception as exc:  # noqa: BLE001 - a bearing must never kill the wake path
            return f', direction unavailable ({exc})'
        if sel is None:
            detail['doa_source'] = 'none'
            return ', direction unavailable (no DOA samples in the utterance window)'
        detail.update({
            'doa_robot_deg': round(math.degrees(sel.azimuth_rad), 1),
            'doa_confidence': round(sel.confidence, 3), 'doa_source': sel.source,
            'doa_n_window': sel.n_window, 'doa_n_active': sel.n_active,
            'doa_n_used': sel.n_used, 'doa_n_distinct': sel.n_distinct,
            'doa_held_deg': (None if sel.held_rad is None
                             else round(math.degrees(sel.held_rad), 1)),
            'doa_n_changed': sel.n_changed, 'doa_held_register': sel.held_register,
        })
        direction = Float64MultiArray()
        direction.data = [float(sel.azimuth_rad), float(sel.confidence)]
        self._dir_pub.publish(direction)
        held = ('' if sel.held_rad is None
                else f', held before {math.degrees(sel.held_rad):+.0f} deg, {sel.n_changed} moved'
                + (', HELD REGISTER' if sel.held_register else ''))
        return (f', direction {math.degrees(sel.azimuth_rad):+.0f} deg (built-in, {sel.source}, '
                f'conf {sel.confidence:.2f}, {sel.n_used}/{sel.n_window} samples, '
                f'{sel.n_distinct} distinct{held})')

    def _tick(self):
        if (self._direction_provider is not None
                and not hasattr(self._direction_provider, 'get_direction_near')):
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
        doa_note = ''
        estimate = getattr(detection, 'doa', None)
        if estimate is not None:
            from come_here_audio.srp_doa import to_robot_frame
            azimuth = to_robot_frame(estimate.azimuth_rad, self._doa_offset_deg, self._doa_mirror)
            detail.update(estimate.as_dict())
            detail['doa_robot_deg'] = round(math.degrees(azimuth), 1)
            firmware_deg = self._read_firmware_doa()
            if firmware_deg is not None:
                detail['doa_firmware_deg'] = firmware_deg
            detail['doa_calibration'] = self._doa_calibration
            # Direction before the wake: the FSM stores it and LISTENING picks it up.
            if self._doa_calibration != 'MISSING':
                direction = Float64MultiArray()
                direction.data = [float(azimuth), float(estimate.confidence)]
                self._dir_pub.publish(direction)
            doa_note = ('' if self._doa_calibration != 'MISSING'
                        else ', DOA UNCALIBRATED: bearing NOT published') + (f', direction {math.degrees(azimuth):+.0f} deg '
                        f'(array {math.degrees(estimate.azimuth_rad):+.0f}, '
                        f'conf {estimate.confidence:.2f}'
                        + (f', firmware {firmware_deg}' if firmware_deg is not None else '')
                        + ')')
        elif self._direction_provider is not None and hasattr(
                self._direction_provider, 'get_direction_near'):
            doa_note = self._utterance_direction(detection, detail)
        elif self._doa_source == 'software':
            doa_note = ', direction unavailable'
        detail_msg = String()
        detail_msg.data = json.dumps(detail)
        self._detail_pub.publish(detail_msg)
        wake = String()
        wake.data = detection.phrase
        self._wake_pub.publish(wake)
        self.get_logger().info(
            f'Wake phrase detected: "{detection.phrase}" (confidence={detection.confidence:.2f}, '
            f'heard="{detection.transcript}", latency={detail["speech_end_to_publish_s"]}s)'
            + doa_note
        )

    def _read_firmware_doa(self):
        """The array's own DOAANGLE (degrees) for the lab log, or None. Never raises."""
        if self._doa_firmware_dev is None:
            return None
        try:
            from come_here_audio import respeaker_tune
            return int(respeaker_tune.read_parameter(self._doa_firmware_dev, 'DOAANGLE'))
        except Exception:  # noqa: BLE001
            return None

    def _health_tick(self):
        health = {'detector': self._detector_name, 'mic': self._mic_label,
                  'doa_calibration': self._doa_calibration}
        if hasattr(self._wake_detector, 'capture_health'):
            health.update(self._wake_detector.capture_health())
        msg = String()
        msg.data = json.dumps(health)
        self._health_pub.publish(msg)
        if health.get('capture_age_s', 0.0) > 2.0:
            self.get_logger().error(f'Microphone capture stalled: {health}')
        if self._doa_calibration == 'MISSING':
            self.get_logger().error(
                'DOA NOT CALIBRATED: the robot will not turn toward the voice. '
                'Run: python3 scripts/calibrate_doa.py')
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
