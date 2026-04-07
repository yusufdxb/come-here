"""Whisper-based wake phrase detector.

Uses OpenAI's Whisper to transcribe audio and detect the "come here"
trigger phrase. Runs inference in a background thread so check() remains
non-blocking.

Supports two backends:
  1. faster-whisper (CTranslate2) -- fast inference, default for base models
  2. HuggingFace transformers -- required when using a LoRA fine-tuned adapter

When adapter_path is set, the HF backend is used automatically.
When adapter_path is None, faster-whisper is used for speed.

Requirements:
  pip install faster-whisper sounddevice numpy
  # For fine-tuned adapter:
  pip install transformers peft torch

The microphone device index is configurable. When the mic hardware is
unknown, pass device=None to use the system default.
"""

import queue
import threading
from typing import Optional

import numpy as np

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
    """Detects "come here" by running Whisper on rolling audio chunks.

    Architecture:
      - A background thread continuously reads audio from the mic
        into fixed-length chunks (default 2s).
      - Each chunk is transcribed with Whisper (faster-whisper or HF).
      - If the transcript contains the trigger phrase, a PhraseDetection
        is queued for the main thread to pick up via check().

    This keeps the ROS spin thread non-blocking.
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
        chunk_duration_s: float = 2.0,
        sample_rate: int = 16000,
        confidence_threshold: float = 0.4,
        no_speech_threshold: float = 0.5,
    ):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._adapter_path = adapter_path
        self._mic_device = mic_device
        self._mic_channels = mic_channels
        self._mic_beam_channel = mic_beam_channel
        self._chunk_duration = chunk_duration_s
        self._sample_rate = sample_rate
        self._confidence_threshold = confidence_threshold
        self._no_speech_threshold = no_speech_threshold

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
        self._detections: queue.Queue[PhraseDetection] = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def setup(self) -> None:
        if self._use_hf:
            self._setup_hf()
        else:
            self._setup_faster_whisper()

        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

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
        # Check for HF cache structure: find the snapshot directory
        snapshots = list(local.rglob("snapshots"))
        if snapshots:
            # Use the first snapshot hash directory
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

    def check(self) -> PhraseDetection | None:
        try:
            return self._detections.get_nowait()
        except queue.Empty:
            return None

    def teardown(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._ct2_model = None
        self._hf_model = None
        self._hf_processor = None

    def _listen_loop(self) -> None:
        """Background thread: record audio chunks and run Whisper on each.

        Records multi-channel audio from the ReSpeaker and extracts the
        beamformed channel (ch0 by default) for Whisper inference.
        """
        import sounddevice as sd

        chunk_samples = int(self._chunk_duration * self._sample_rate)

        while self._running:
            try:
                audio = sd.rec(
                    chunk_samples,
                    samplerate=self._sample_rate,
                    channels=self._mic_channels,
                    dtype='float32',
                    device=self._mic_device,
                )
                sd.wait()

                # Extract selected channel and apply gain boost
                audio_np = audio[:, self._mic_beam_channel].flatten().astype(np.float32)
                audio_np = np.clip(audio_np * 4.0, -1.0, 1.0)

                # Skip silent chunks (threshold accounts for gain)
                if np.max(np.abs(audio_np)) < 0.02:
                    continue

                if self._use_hf:
                    self._transcribe_hf(audio_np)
                else:
                    self._transcribe_ct2(audio_np)

            except Exception:
                # Don't crash the thread on transient audio errors
                if self._running:
                    import time
                    time.sleep(0.5)

    def _transcribe_ct2(self, audio_np: np.ndarray) -> None:
        """Transcribe using faster-whisper (CTranslate2).

        Note: vad_filter is disabled — it crashes on ARM/Jetson due to
        an onnxruntime assertion failure. We use the ReSpeaker's hardware
        VAD (VOICEACTIVITY register) and silence detection instead.
        """
        segments, info = self._ct2_model.transcribe(
            audio_np,
            beam_size=1,
            language="en",
        )

        for segment in segments:
            # Filter hallucinations from silence/noise
            if segment.no_speech_prob > self._no_speech_threshold:
                continue

            text = segment.text.strip().lower()
            avg_logprob = segment.avg_log_prob
            confidence = min(1.0, max(0.0, 1.0 + avg_logprob))

            if confidence < self._confidence_threshold:
                continue

            for trigger in self.TRIGGER_PHRASES:
                if trigger in text:
                    self._detections.put(
                        PhraseDetection(phrase=trigger, confidence=confidence)
                    )
                    break

    def _transcribe_hf(self, audio_np: np.ndarray) -> None:
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

        # HF doesn't give per-segment log probs as easily, use a fixed confidence
        # when text matches (the fine-tuned model's accuracy is the real signal)
        confidence = 0.85

        for trigger in self.TRIGGER_PHRASES:
            if trigger in text:
                self._detections.put(
                    PhraseDetection(phrase=trigger, confidence=confidence)
                )
                break
