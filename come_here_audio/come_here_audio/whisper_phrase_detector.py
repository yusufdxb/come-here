"""Whisper-based wake phrase detector with streaming capture.

Uses a three-stage pipeline for low-latency wake phrase detection:
  1. InputStream callback -> RingBuffer (continuous, non-blocking)
  2. Segmenter thread: energy VAD -> endpoint detection -> LatestOnlyQueue
  3. Inference thread: Whisper transcription -> detection callback

Supports two backends:
  1. faster-whisper (CTranslate2) -- fast inference, default for base models
  2. HuggingFace transformers -- required when using a LoRA fine-tuned adapter

When adapter_path is set, the HF backend is used automatically.
When adapter_path is None, faster-whisper is used for speed.

Requirements:
  pip install faster-whisper sounddevice numpy
  # For fine-tuned adapter:
  pip install transformers peft torch
"""

import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

from come_here_audio.ring_buffer import LatestOnlyQueue, RingBuffer
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


class WhisperPhraseDetector(WakePhraseDetector):
    """Detects "come here" using streaming audio capture and Whisper.

    Architecture:
      - InputStream callback writes mic audio into a RingBuffer continuously.
      - Segmenter thread hops over the buffer, detecting speech endpoints
        via energy-based VAD: SILENCE -> SPEECH -> TRAILING_SILENCE.
      - On endpoint, the speech window is pushed to a LatestOnlyQueue
        (stale segments are auto-dropped).
      - Inference thread pulls from the queue, runs Whisper, and fires
        the detection callback if "come here" is found.

    The old check()-based API is preserved for backward compatibility.
    For low-latency use, register a callback via set_on_detection().
    """

    TRIGGER_PHRASES = {"come here", "come over here"}

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        adapter_path: Optional[str] = None,
        mic_device: Optional[str] = None,
        mic_channels: int = 6,
        mic_beam_channel: int = 1,
        sample_rate: int = 16000,
        confidence_threshold: float = 0.4,
        no_speech_threshold: float = 0.5,
        # Streaming params
        mic_gain: float = 4.0,
        window_duration_s: float = 1.0,
        hop_duration_ms: int = 250,
        end_silence_ms: int = 200,
        ring_buffer_duration_s: float = 3.0,
        energy_threshold: float = 0.001,
        vad_check_fn: Optional[Callable[[], bool]] = None,
    ):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._adapter_path = adapter_path
        self._mic_device = mic_device
        self._mic_channels = mic_channels
        self._mic_beam_channel = mic_beam_channel
        self._sample_rate = sample_rate
        self._confidence_threshold = confidence_threshold
        self._no_speech_threshold = no_speech_threshold
        self._mic_gain = mic_gain
        self._window_duration_s = window_duration_s
        self._hop_duration_ms = hop_duration_ms
        self._end_silence_ms = end_silence_ms
        self._ring_buffer_duration_s = ring_buffer_duration_s
        self._energy_threshold = energy_threshold
        self._vad_check_fn = vad_check_fn

        # Determine backend
        self._use_hf = adapter_path is not None

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

        self._ct2_model: Optional["WhisperModel"] = None
        self._hf_model = None
        self._hf_processor = None

        # Backward-compat detection queue (for check())
        self._detections: queue.Queue[PhraseDetection] = queue.Queue()
        # Event-driven callback
        self._on_detection_cb: Optional[Callable] = None

        self._running = False
        self._stream = None
        self._ring_buffer: Optional[RingBuffer] = None
        self._segment_queue: Optional[LatestOnlyQueue] = None
        self._segmenter_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        # Cooldown: suppress duplicate detections from overlapping windows
        self._last_detection_time: float = 0.0
        self._detection_cooldown_s: float = 3.0
        # Highpass filter state (initialized in _start_audio_stream)
        self._hp_sos = None
        self._hp_zi = None

    def set_on_detection(self, callback: Callable[[PhraseDetection, float], None]) -> None:
        """Register a callback for event-driven wake detection.

        Args:
            callback: Called with (detection, t_speech_end) when a phrase
                is detected. t_speech_end is the monotonic timestamp of
                the speech endpoint.
        """
        self._on_detection_cb = callback

    def setup(self) -> None:
        if self._use_hf:
            self._setup_hf()
        else:
            self._setup_faster_whisper()

        # Allocate ring buffer
        buf_samples = int(self._ring_buffer_duration_s * self._sample_rate)
        self._ring_buffer = RingBuffer(capacity=buf_samples)
        self._segment_queue = LatestOnlyQueue()

        self._running = True

        # Start audio input stream
        self._start_audio_stream()

        # Start segmenter and inference threads
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

        # Highpass filter at 300Hz to remove motor noise rumble.
        # Motor vibration is concentrated below 200Hz; voice formants
        # are 300-3000Hz. This dramatically improves SNR on the GO2.
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
        """PortAudio callback: extract beam channel, filter, apply gain, write to ring buffer.

        Runs in PortAudio's thread — keep it fast, no allocations beyond the slice.
        """
        if not self._running:
            return
        # Extract the selected channel
        mono = indata[:, self._mic_beam_channel].copy()
        # Highpass filter: remove motor noise below 300Hz
        if self._hp_sos is not None:
            from scipy.signal import sosfilt
            mono, self._hp_zi = sosfilt(self._hp_sos, mono, zi=self._hp_zi)
        # Apply gain and clip
        mono = mono * self._mic_gain
        np.clip(mono, -1.0, 1.0, out=mono)
        self._ring_buffer.write(mono)

    def _segmenter_loop(self) -> None:
        """Emit audio segments for Whisper inference, gated by hardware VAD.

        When a vad_check_fn is provided (from ReSpeaker hardware VAD),
        segments are only queued when VAD indicates recent speech. This
        prevents wasting inference time on noise, keeping the inference
        thread free for actual speech — cutting latency significantly.

        Without vad_check_fn, falls back to peak-amplitude gating.

        The LatestOnlyQueue ensures that if inference is slower than the
        hop interval, stale segments are dropped automatically.
        """
        window_samples = int(self._window_duration_s * self._sample_rate)
        hop_s = self._hop_duration_ms / 1000.0
        # Minimum samples before first emission (wait for buffer to fill)
        min_written = window_samples

        while self._running:
            time.sleep(hop_s)

            if self._ring_buffer.total_written < min_written:
                continue

            segment = self._ring_buffer.read_last(window_samples)
            peak = float(np.max(np.abs(segment)))

            # Skip truly dead silence (below noise floor)
            if peak < 0.02:
                continue

            # VAD gating: ALWAYS check hardware VAD when available.
            # Motor noise on the GO2 produces peaks of 0.6-0.9 which
            # would bypass any peak-based threshold. Only queue when
            # the ReSpeaker firmware detects actual voice activity.
            if self._vad_check_fn is not None:
                if not self._vad_check_fn():
                    continue
                print(f"[SEG] VAD-gated segment queued (peak={peak:.3f})")
            else:
                # No hardware VAD — fall back to peak threshold
                if peak < 0.10:
                    continue
                print(f"[SEG] Peak-gated segment queued (peak={peak:.3f})")

            t_speech_end = time.monotonic()
            self._segment_queue.put((segment, t_speech_end))

    def _inference_loop(self) -> None:
        """Pull segments from queue, run Whisper, fire callbacks on match."""
        while self._running:
            try:
                segment, t_speech_end = self._segment_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Skip near-silent segments
            peak = float(np.max(np.abs(segment)))
            if peak < 0.02:
                continue

            rms = float(np.sqrt(np.mean(segment ** 2)))

            t_infer_start = time.monotonic()

            if self._use_hf:
                detection = self._transcribe_hf(segment)
            else:
                detection = self._transcribe_ct2(segment)

            t_infer_done = time.monotonic()
            infer_ms = (t_infer_done - t_infer_start) * 1000

            if detection is not None:
                # Cooldown: suppress duplicates from overlapping windows
                now = time.monotonic()
                if now - self._last_detection_time < self._detection_cooldown_s:
                    print(f"[WHISPER] Suppressed duplicate: '{detection.phrase}' "
                          f"(cooldown {self._detection_cooldown_s:.0f}s)")
                    continue
                self._last_detection_time = now

                print(f"[WHISPER] MATCH: '{detection.phrase}' "
                      f"conf={detection.confidence:.2f} "
                      f"peak={peak:.3f} rms={rms:.4f} "
                      f"infer={infer_ms:.0f}ms")
                # Backward compat: put in queue for check()
                self._detections.put(detection)
                # Event-driven: fire callback
                if self._on_detection_cb is not None:
                    self._on_detection_cb(detection, t_speech_end)
            else:
                # Log non-matching transcriptions for calibration
                print(f"[WHISPER] no match | peak={peak:.3f} rms={rms:.4f} "
                      f"infer={infer_ms:.0f}ms")

    def check(self) -> PhraseDetection | None:
        """Backward-compatible polling interface."""
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

    def _transcribe_ct2(self, audio_np: np.ndarray) -> PhraseDetection | None:
        """Transcribe using faster-whisper (CTranslate2)."""
        segments, info = self._ct2_model.transcribe(
            audio_np,
            beam_size=1,
            language="en",
            initial_prompt="come here",
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

            # Log all passing transcriptions (matched or not)
            for trigger in self.TRIGGER_PHRASES:
                if trigger in text:
                    return PhraseDetection(phrase=trigger, confidence=confidence)

            # Passed thresholds but didn't match trigger phrase
            if text:
                print(f"[WHISPER] heard: '{text}' conf={confidence:.2f} "
                      f"(no trigger match)")

        if seg_count == 0:
            print(f"[WHISPER] empty (no segments produced)")

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

        for trigger in self.TRIGGER_PHRASES:
            if trigger in text:
                return PhraseDetection(phrase=trigger, confidence=confidence)

        return None
