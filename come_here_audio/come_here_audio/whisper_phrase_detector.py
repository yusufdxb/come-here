"""Whisper-based wake phrase detector with streaming capture and a far-field front end.

Pipeline:
  1. InputStream callback -> RingBuffer (continuous, non-blocking)
  2. Segmenter thread: every captured 50 ms frame, in order, passes an energy
     gate; an utterance keeps a short pre-roll from before its onset and
     closes on trailing silence -> LatestOnlyQueue
  3. Inference thread: Whisper transcription -> fuzzy "come here" match

Far-field front end:
  * Adaptive gate. A fixed RMS threshold is an absolute loudness test, so a
    talker a few metres away fails it in a quiet room while a fan passes it in
    a noisy one. The gate is max(gate_floor_rms, noise_floor * gate_snr_margin),
    where the noise floor is the 25th percentile of recent frames captured
    outside an utterance. gate_floor_rms keeps silence from ever opening it,
    and Whisper's own VAD (whisper_vad_filter) is the second opinion.
  * Bootstrap. The room is measured for calibration_s before listening, and
    re-measured whenever an utterance runs to max_utterance_sec. Without both,
    a room whose ambient sits above gate_floor_rms (measured 0.0054 on a
    ReSpeaker beam channel against a 0.003 floor) never lets the floor learn,
    and the room segments itself into endless utterances.
  * preroll_sec of audio from before the onset reaches Whisper, so the first
    consonant of "come" is not cut off.
  * An utterance needs min_utterance_sec of voiced audio: trailing silence
    cannot turn a click into speech.
  * CTranslate2 is capped at cpu_threads. 0 means every core, which starved the
    camera pipeline when voice and vision shared the Orin.

Supports two backends:
  1. faster-whisper (CTranslate2) -- fast inference, default for base models
  2. HuggingFace transformers -- required when using a LoRA fine-tuned adapter

Requirements:
  pip install faster-whisper sounddevice numpy
  # For fine-tuned adapter:
  pip install transformers peft torch
"""

import collections
import math
import queue
import threading
import time
from typing import Callable, Optional, Sequence

import numpy as np

from come_here_audio.come_here_matcher import match_come_here, match_good_boy
from come_here_audio.phrase_matcher import match_trigger
from come_here_audio.ring_buffer import LatestOnlyQueue, MultiRingBuffer, RingBuffer
from come_here_audio.wake_phrase_detector import PhraseDetection, WakePhraseDetector

# Guarded imports
try:
    from faster_whisper import WhisperModel
    _FASTER_WHISPER_AVAILABLE = True
except ImportError:
    _FASTER_WHISPER_AVAILABLE = False

try:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

FRAME_SEC = 0.05


class WhisperPhraseDetector(WakePhraseDetector):
    """Detects "come here" using streaming audio capture and Whisper.

    The check()-based polling API is the interface audio_node uses. For
    event-driven use, register a callback via set_on_detection().
    """

    TRIGGER_PHRASES = {'come here'}  # lab 09-14: 'come over here' matched 'go over here' (false wake); 48/57 hits, 0/14 false

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        adapter_path: Optional[str] = None,
        mic_device=None,
        mic_channels: int = 1,
        mic_beam_channel: int = 0,
        sample_rate: int = 16000,
        confidence_threshold: float = 0.4,
        no_speech_threshold: float = 0.5,
        phrase_ratio_threshold: float = 0.80,
        mic_gain: float = 1.0,
        highpass_filter: bool = False,
        # Software DOA: an estimator with .estimate(samples x mics) and the
        # capture channels that carry the raw capsules (ReSpeaker: 1..4).
        doa_estimator=None,
        doa_channels: Sequence[int] = (1, 2, 3, 4),
        ring_buffer_duration_s: float = 6.0,
        # Utterance endpointing
        utterance_rms_threshold: float = 0.015,
        silence_to_end_sec: float = 0.70,
        min_utterance_sec: float = 0.20,
        max_utterance_sec: float = 4.50,
        vad_check_fn: Optional[Callable[[], bool]] = None,
        # Far-field front end
        adaptive_gate: bool = True,
        gate_snr_margin: float = 2.0,
        gate_floor_rms: float = 0.003,
        noise_window_s: float = 3.0,
        calibration_s: float = 1.0,
        preroll_sec: float = 0.25,
        segment_peak_threshold: float = 0.02,
        whisper_vad_filter: bool = True,
        cpu_threads: int = 2,
        cooldown_s: float = 3.0,
        # Also listen for "good boy" (praise: stands a seated robot back up).
        praise_enabled: bool = False,
        # Deprecated, fixed-hop segmenter was replaced by utterance endpointing.
        # Kept so existing callers (e.g. hear_and_rotate_demo) don't raise TypeError.
        window_duration_s: Optional[float] = None,
        hop_duration_ms: Optional[int] = None,
        end_silence_ms: Optional[int] = None,
        energy_threshold: Optional[float] = None,
    ):
        if sample_rate != 16000:
            raise ValueError('Whisper capture requires sample_rate=16000')
        if not 0 <= mic_beam_channel < mic_channels:
            raise ValueError('mic_beam_channel must be within mic_channels')
        doa_channels = [int(c) for c in doa_channels]
        if doa_estimator is not None:
            if any(not 0 <= c < mic_channels for c in doa_channels):
                raise ValueError('doa_channels must be within mic_channels')
            n_mics = getattr(doa_estimator, 'n_mics', len(doa_channels))
            if n_mics != len(doa_channels):
                raise ValueError(f'doa_estimator expects {n_mics} mics, got {len(doa_channels)} channels')
        for name, value in (
            ('utterance_rms_threshold', utterance_rms_threshold),
            ('silence_to_end_sec', silence_to_end_sec),
            ('min_utterance_sec', min_utterance_sec),
            ('max_utterance_sec', max_utterance_sec),
            ('mic_gain', mic_gain),
            ('gate_snr_margin', gate_snr_margin),
            ('noise_window_s', noise_window_s),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name, value in (
            ('preroll_sec', preroll_sec),
            ('segment_peak_threshold', segment_peak_threshold),
            ('gate_floor_rms', gate_floor_rms),
            ('calibration_s', calibration_s),
            ('cooldown_s', cooldown_s),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if max_utterance_sec < min_utterance_sec:
            raise ValueError('max_utterance_sec must be >= min_utterance_sec')

        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._cpu_threads = int(cpu_threads)
        self._adapter_path = adapter_path
        self._mic_device = mic_device
        self._mic_channels = mic_channels
        self._mic_beam_channel = mic_beam_channel
        self._doa_estimator = doa_estimator
        self._doa_channels = doa_channels
        self._raw_ring: Optional[MultiRingBuffer] = None
        self._last_doa = None
        self._sample_rate = sample_rate
        self._confidence_threshold = confidence_threshold
        self._no_speech_threshold = no_speech_threshold
        self._phrase_ratio_threshold = phrase_ratio_threshold
        self._mic_gain = mic_gain
        self._highpass_filter = highpass_filter
        self._ring_buffer_duration_s = ring_buffer_duration_s
        self._utterance_rms_threshold = utterance_rms_threshold
        self._silence_to_end_sec = silence_to_end_sec
        self._min_utterance_sec = min_utterance_sec
        self._max_utterance_sec = max_utterance_sec
        self._vad_check_fn = vad_check_fn
        self._adaptive_gate = bool(adaptive_gate)
        self._gate_snr_margin = float(gate_snr_margin)
        self._gate_floor_rms = float(gate_floor_rms)
        self._segment_peak_threshold = float(segment_peak_threshold)
        self._whisper_vad_filter = bool(whisper_vad_filter)
        self._preroll_sec = float(preroll_sec)

        self._frame_samples = int(FRAME_SEC * sample_rate)
        window_frames = max(8, int(noise_window_s / FRAME_SEC))
        # RMS of frames captured outside an utterance: the only honest estimate
        # of what this room and this robot sound like.
        self._noise_frames = collections.deque(maxlen=window_frames)
        # RMS of every recent frame, for re-measuring a room that rose above the gate.
        self._recent_levels = collections.deque(maxlen=window_frames)
        self._noise_floor = 0.0
        self._calibration_frames = int(round(calibration_s / FRAME_SEC))
        self._frames_seen = 0

        # Determine backend
        self._use_hf = adapter_path is not None

        self._ct2_model: Optional["WhisperModel"] = None
        self._hf_model = None
        self._hf_processor = None

        # Polling detection queue (for check())
        self._detections: queue.Queue[PhraseDetection] = queue.Queue()
        # Event-driven callback
        self._on_detection_cb: Optional[Callable] = None

        self._running = False
        self._stream = None
        self._ring_buffer: Optional[RingBuffer] = None
        self._segment_queue = LatestOnlyQueue()
        self._segmenter_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        # Cooldown: suppress duplicate detections of one utterance
        self._last_detection_time: float = 0.0
        self._detection_cooldown_s: float = float(cooldown_s)
        self._praise_enabled = bool(praise_enabled)
        # Highpass filter state (initialized in _start_audio_stream)
        self._hp_sos = None
        self._hp_zi = None
        # Capture evidence for health reporting
        self._last_capture_s = 0.0
        self._capture_rms = 0.0
        self._capture_peak = 0.0
        self._capture_status_count = 0
        self._reset_segmenter()

    def set_on_detection(self, callback: Callable[[PhraseDetection, float], None]) -> None:
        """Register a callback called with (detection, t_speech_end)."""
        self._on_detection_cb = callback

    def setup(self) -> None:
        if self._use_hf and not _HF_AVAILABLE:
            raise ImportError(
                "transformers, peft, and torch are required for fine-tuned adapter. "
                "Install with: pip install transformers peft torch"
            )
        if not self._use_hf and not _FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper is required for WhisperPhraseDetector. "
                "Install with: pip install faster-whisper"
            )
        if self._use_hf:
            self._setup_hf()
        else:
            self._setup_faster_whisper()

        buf_samples = int(self._ring_buffer_duration_s * self._sample_rate)
        self._ring_buffer = RingBuffer(capacity=buf_samples)
        if self._doa_estimator is not None:
            self._raw_ring = MultiRingBuffer(buf_samples, len(self._doa_channels))
        self._segment_queue = LatestOnlyQueue()
        self._reset_segmenter()

        self._running = True
        self._start_audio_stream()

        self._segmenter_thread = threading.Thread(
            target=self._segmenter_loop, daemon=True, name="segmenter"
        )
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="inference"
        )
        self._segmenter_thread.start()
        self._inference_thread.start()

    def _start_audio_stream(self) -> None:
        """Open a non-blocking InputStream for continuous capture."""
        import sounddevice as sd

        # Highpass filter at 300Hz, off by default. The ReSpeaker DSP channel
        # (ch 0) already runs noise suppression; stacking a 300Hz HP on top of
        # it attenuates lower voice formants for no real gain. Only enable when
        # bypassing the DSP (e.g. reading a raw capsule directly).
        if self._highpass_filter:
            try:
                from scipy.signal import butter, sosfilt_zi
                self._hp_sos = butter(4, 300, btype='highpass',
                                      fs=self._sample_rate, output='sos')
                self._hp_zi = sosfilt_zi(self._hp_sos) * 0.0
                print("[AUDIO] Highpass filter at 300Hz enabled")
            except ImportError:
                print("[AUDIO] scipy not available, no highpass filter")

        blocksize = int(0.1 * self._sample_rate)  # 100ms blocks
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._mic_channels,
            dtype="float32",
            device=self._mic_device,
            blocksize=blocksize,
            callback=self._audio_callback,
        )
        self._stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        """PortAudio callback: pick channel, optionally filter/gain, write to ring buffer.

        Runs in PortAudio's thread, keep it fast. With defaults
        (mic_channels=1, mic_beam_channel=0) this reads the ReSpeaker DSP
        output (beamformer + AEC + AGC + NS applied in firmware) straight through.
        """
        if not self._running:
            return
        self._last_capture_s = time.monotonic()
        if status:
            self._capture_status_count += 1
            print(f'[AUDIO] capture status: {status}')
        if self._raw_ring is not None:
            # Same frames, same count: absolute positions line up with the mono ring.
            self._raw_ring.write(indata[:, self._doa_channels])
        mono = indata[:, self._mic_beam_channel].copy()
        if self._hp_sos is not None:
            from scipy.signal import sosfilt
            mono, self._hp_zi = sosfilt(self._hp_sos, mono, zi=self._hp_zi)
        if self._mic_gain != 1.0:
            mono = mono * self._mic_gain
            np.clip(mono, -1.0, 1.0, out=mono)
        self._capture_rms = float(np.sqrt(np.mean(mono ** 2)))
        self._capture_peak = float(np.max(np.abs(mono)))
        self._ring_buffer.write(mono)

    # -- gate --

    def effective_rms_threshold(self) -> float:
        """What the segmenter is gating on right now."""
        if not self._adaptive_gate:
            return self._utterance_rms_threshold
        return max(self._gate_floor_rms, self._noise_floor * self._gate_snr_margin)

    def effective_peak_threshold(self) -> float:
        """The segment peak gate, kept in the configured ratio to the RMS gate."""
        if not self._adaptive_gate:
            return self._segment_peak_threshold
        ratio = self._segment_peak_threshold / self._utterance_rms_threshold
        return self.effective_rms_threshold() * ratio

    def _update_noise_floor(self, rms: float) -> None:
        self._noise_frames.append(rms)
        if len(self._noise_frames) >= 8:
            # A low percentile, not the mean: a window that caught an onset is
            # not a noise floor.
            self._noise_floor = float(np.percentile(np.asarray(self._noise_frames), 25))

    def _recalibrate_noise_floor(self) -> None:
        """Re-measure the floor from raw recent levels, ignoring the current gate."""
        if not self._recent_levels:
            return
        self._noise_frames.clear()
        self._noise_frames.extend(self._recent_levels)
        self._noise_floor = float(np.percentile(np.asarray(self._noise_frames), 25))

    def capture_health(self) -> dict:
        """Latest callback evidence, separate from merely opening a stream."""
        return {
            'capture_age_s': round(time.monotonic() - self._last_capture_s, 2),
            'rms': round(self._capture_rms, 5),
            'peak': round(self._capture_peak, 5),
            'noise_floor': round(self._noise_floor, 5),
            'rms_gate': round(self.effective_rms_threshold(), 5),
            'peak_gate': round(self.effective_peak_threshold(), 5),
            'adaptive_gate': self._adaptive_gate,
            'calibrated': self._frames_seen >= self._calibration_frames,
            'status_count': self._capture_status_count,
            'active': bool(self._stream is not None and self._stream.active),
        }

    # -- segmenter --

    def _reset_segmenter(self) -> None:
        preroll_frames = max(1, math.ceil(self._preroll_sec / FRAME_SEC))
        self._seg_preroll = collections.deque(maxlen=preroll_frames)
        self._seg_pending = np.empty(0, dtype=np.float32)
        self._seg_position = 0
        self._seg_frame_end = 0
        self._seg_utterance = []
        self._seg_utterance_start = 0
        self._seg_last_speech = 0

    def _segmenter_loop(self) -> None:
        while self._running:
            time.sleep(0.025)
            captured, end = self._ring_buffer.read_since(self._seg_position)
            self._process_capture(captured, end, time.monotonic())

    def _process_capture(self, captured: np.ndarray, end: int, now: float) -> None:
        """Segment newly captured samples; ``end`` is the total written after them."""
        frame_samples = self._frame_samples
        min_samples = int(self._min_utterance_sec * self._sample_rate)
        max_samples = int(self._max_utterance_sec * self._sample_rate)
        silence_needed = int(self._silence_to_end_sec * self._sample_rate)

        if end - len(captured) > self._seg_position:
            print('[AUDIO] capture overrun: discarding incomplete utterance')
            self._seg_pending = np.empty(0, dtype=np.float32)
            self._seg_utterance = []
            self._seg_preroll.clear()
            self._seg_frame_end = end - len(captured)
        self._seg_position = end
        if len(captured) == 0:
            return

        pending = np.concatenate((self._seg_pending, captured))
        offset = 0
        while len(pending) - offset >= frame_samples:
            frame = pending[offset:offset + frame_samples].copy()
            offset += frame_samples
            self._seg_frame_end += frame_samples
            rms = float(np.sqrt(np.mean(frame ** 2)))
            self._recent_levels.append(rms)
            self._frames_seen += 1

            if self._adaptive_gate and self._frames_seen <= self._calibration_frames:
                # Measure the room before listening to it.
                self._update_noise_floor(rms)
                self._seg_preroll.append(frame)
                continue

            voiced = rms >= self.effective_rms_threshold()
            if not self._seg_utterance:
                if self._adaptive_gate:
                    # Every frame outside an utterance describes the room. Taking
                    # only frames already under the gate cannot bootstrap.
                    self._update_noise_floor(rms)
                if voiced:
                    self._seg_utterance = list(self._seg_preroll) + [frame]
                    self._seg_utterance_start = self._seg_frame_end - frame_samples
                    self._seg_last_speech = self._seg_frame_end
            else:
                self._seg_utterance.append(frame)
                if voiced:
                    self._seg_last_speech = self._seg_frame_end
                silence_len = self._seg_frame_end - self._seg_last_speech
                utt_len = self._seg_frame_end - self._seg_utterance_start
                hit_max = utt_len >= max_samples
                if silence_len >= silence_needed or hit_max:
                    # Trailing silence must not make a click qualify as speech.
                    if self._seg_last_speech - self._seg_utterance_start >= min_samples:
                        segment = np.concatenate(self._seg_utterance)
                        vad_note = ''
                        if self._vad_check_fn is not None:
                            try:
                                vad_hit = self._vad_check_fn()
                                vad_note = f" vad={'hit' if vad_hit else 'miss (advisory)'}"
                            except Exception as exc:  # noqa: BLE001
                                vad_note = f' vad=unavailable ({exc})'
                        print(f'[SEG] utterance closed '
                              f'len={len(segment) / self._sample_rate:.2f}s '
                              f'gate={self.effective_rms_threshold():.4f}{vad_note}')
                        speech_end_s = now - (end - self._seg_last_speech) / self._sample_rate
                        speech_start_s = now - (end - self._seg_utterance_start) / self._sample_rate
                        self._segment_queue.put((
                            segment, speech_end_s,
                            (self._seg_utterance_start, self._seg_last_speech),
                            speech_start_s,
                        ))
                    if hit_max and self._adaptive_gate:
                        # Ran to max length: the room itself is above the gate.
                        self._recalibrate_noise_floor()
                    self._seg_utterance = []
                    self._seg_preroll.clear()
            self._seg_preroll.append(frame)
        self._seg_pending = pending[offset:].copy()

    # -- inference --

    def _inference_loop(self) -> None:
        """Pull segments from queue, run Whisper, fire callbacks on match."""
        while self._running:
            try:
                item = self._segment_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            segment, t_speech_end, span = item[:3]
            t_speech_start = item[3] if len(item) > 3 else None

            peak = float(np.max(np.abs(segment)))
            peak_threshold = self.effective_peak_threshold()
            if peak < peak_threshold:
                print(f'[WHISPER] rejected peak={peak:.4f} threshold={peak_threshold:.4f} '
                      f'(noise floor {self._noise_floor:.4f})')
                continue

            rms = float(np.sqrt(np.mean(segment ** 2)))
            # Direction first: cheap, and it belongs to exactly these samples.
            doa = self._estimate_doa(span)
            t_infer_start = time.monotonic()
            if self._use_hf:
                detection = self._transcribe_hf(segment)
            else:
                detection = self._transcribe_ct2(segment)
            infer_ms = (time.monotonic() - t_infer_start) * 1000

            if detection is not None:
                now = time.monotonic()
                if now - self._last_detection_time < self._detection_cooldown_s:
                    print(f"[WHISPER] Suppressed duplicate: '{detection.phrase}' "
                          f"(cooldown {self._detection_cooldown_s:.0f}s)")
                    continue
                self._last_detection_time = now
                detection.t_speech_end = t_speech_end
                detection.t_speech_start = t_speech_start
                detection.infer_ms = infer_ms
                detection.doa = doa

                print(f"[WHISPER] MATCH: '{detection.phrase}' "
                      f"conf={detection.confidence:.2f} "
                      f"peak={peak:.3f} rms={rms:.4f} "
                      f"infer={infer_ms:.0f}ms")
                self._detections.put(detection)
                if self._on_detection_cb is not None:
                    self._on_detection_cb(detection, t_speech_end)
            else:
                print(f"[WHISPER] no match | peak={peak:.3f} rms={rms:.4f} "
                      f"infer={infer_ms:.0f}ms")

    def _estimate_doa(self, span):
        """Direction of the utterance at ring positions ``span``; None without an estimator."""
        if self._doa_estimator is None or self._raw_ring is None or span is None:
            return None
        start, end = int(span[0]), int(span[1])
        raw = self._raw_ring.read_range(start, end)
        if len(raw) < end - start:
            print(f'[DOA] raw ring lost {end - start - len(raw)} samples of the utterance')
        try:
            est = self._doa_estimator.estimate(raw)
        except Exception as exc:  # noqa: BLE001 - a bearing must never kill the wake path
            print(f'[DOA] estimate failed: {exc}')
            return None
        if est is None:
            print('[DOA] no estimate: clip too short or silent')
        else:
            print(f'[DOA] array {math.degrees(est.azimuth_rad):+.0f} deg '
                  f'conf={est.confidence:.2f} peak={est.peak:.3f} '
                  f'contrast={est.contrast:.2f} frames={est.frames_used}')
        self._last_doa = est
        return est

    @property
    def last_doa(self):
        """Most recent software DOA estimate (any utterance, matched or not)."""
        return self._last_doa

    def check(self) -> PhraseDetection | None:
        """Polling interface."""
        try:
            return self._detections.get_nowait()
        except queue.Empty:
            return None

    def teardown(self) -> None:
        self._running = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._segmenter_thread is not None:
            self._segmenter_thread.join(timeout=3.0)
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=3.0)

        self._ct2_model = None
        self._hf_model = None
        self._hf_processor = None

    # --- Model setup ---

    def _resolve_local_model(self, subdir: str) -> str | None:
        """Check if a local model cache exists alongside the package.

        Handles both flat layout (models/faster-whisper-base.en/*.bin)
        and HuggingFace cache layout (models/.../snapshots/<hash>/*.bin).
        """
        import pathlib
        pkg_dir = pathlib.Path(__file__).resolve().parent.parent.parent
        local = pkg_dir / "models" / subdir
        if not local.is_dir():
            return None
        snapshots = list(local.rglob("snapshots"))
        if snapshots:
            snap_dirs = list(snapshots[0].iterdir())
            if snap_dirs:
                return str(snap_dirs[0])
        return str(local)

    def _setup_faster_whisper(self) -> None:
        local = self._resolve_local_model(f"faster-whisper-{self._model_size}")
        model_path = local if local else self._model_size
        self._ct2_model = WhisperModel(
            model_path,
            device=self._device,
            compute_type=self._compute_type,
            cpu_threads=self._cpu_threads,
        )

    def _setup_hf(self) -> None:
        local = self._resolve_local_model(f"whisper-{self._model_size}")
        hf_model_name = local if local else f"openai/whisper-{self._model_size}"
        self._hf_processor = WhisperProcessor.from_pretrained(hf_model_name)

        hf_device = self._device
        if hf_device == "cpu":
            dtype = torch.float32
        else:
            dtype = torch.float16

        model = WhisperForConditionalGeneration.from_pretrained(
            hf_model_name, torch_dtype=dtype
        )
        model = PeftModel.from_pretrained(model, self._adapter_path)
        model = model.merge_and_unload()
        model = model.to(hf_device)
        model.eval()
        self._hf_model = model

    # --- Transcription ---

    def _match(self, text: str):
        """Trigger match for one transcript.

        {"come here"} uses the bounded whole-token matcher (come_here_matcher:
        no substring hits such as "welcome here everyone"); any other trigger
        set keeps the generic substring + difflib matcher. With praise_enabled,
        a transcript with no wake match is also tried against "good boy".
        """
        if set(self.TRIGGER_PHRASES) == {'come here'}:
            match = match_come_here(text)
        else:
            match = match_trigger(text, self.TRIGGER_PHRASES,
                                  ratio_threshold=self._phrase_ratio_threshold)
        if match is None and getattr(self, '_praise_enabled', False):
            match = match_good_boy(text)
        return match

    def _transcribe_ct2(self, audio_np: np.ndarray) -> PhraseDetection | None:
        """Transcribe using faster-whisper (CTranslate2)."""
        segments, info = self._ct2_model.transcribe(
            audio_np,
            beam_size=1,
            language="en",
            initial_prompt="come here",
            vad_filter=self._whisper_vad_filter,
        )

        seg_count = 0
        for segment in segments:
            seg_count += 1
            text = segment.text.strip().lower()
            avg_logprob = segment.avg_logprob
            confidence = min(1.0, max(0.0, 1.0 + avg_logprob))

            if segment.no_speech_prob > self._no_speech_threshold:
                print(f"[WHISPER] rejected no_speech: '{text}' "
                      f"no_speech={segment.no_speech_prob:.2f}")
                continue

            if confidence < self._confidence_threshold:
                print(f"[WHISPER] rejected low_conf: '{text}' "
                      f"conf={confidence:.2f} logprob={avg_logprob:.2f}")
                continue

            match = self._match(text)
            if match is not None:
                if match.ratio < 1.0:
                    print(f"[WHISPER] fuzzy match: heard '{match.heard}' "
                          f"~ '{match.phrase}' ratio={match.ratio:.2f}")
                return PhraseDetection(
                    phrase=match.phrase, confidence=confidence,
                    transcript=text, ratio=match.ratio,
                )

            if text:
                print(f"[WHISPER] heard: '{text}' conf={confidence:.2f} "
                      f"(no trigger match)")

        if seg_count == 0:
            print("[WHISPER] empty (no segments produced)")

        return None

    def _transcribe_hf(self, audio_np: np.ndarray) -> PhraseDetection | None:
        """Transcribe using HuggingFace transformers (supports LoRA adapter)."""
        inputs = self._hf_processor.feature_extractor(
            audio_np, sampling_rate=self._sample_rate, return_tensors="pt"
        )

        device = next(self._hf_model.parameters()).device
        dtype = next(self._hf_model.parameters()).dtype
        input_features = inputs.input_features.to(device, dtype=dtype)

        with torch.no_grad():
            predicted_ids = self._hf_model.generate(input_features, max_new_tokens=30)

        text = self._hf_processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip().lower()

        confidence = 0.85

        match = self._match(text)
        if match is not None:
            if match.ratio < 1.0:
                print(f"[WHISPER] fuzzy match (HF): heard '{match.heard}' "
                      f"~ '{match.phrase}' ratio={match.ratio:.2f}")
            return PhraseDetection(
                phrase=match.phrase, confidence=confidence,
                transcript=text, ratio=match.ratio,
            )

        return None
