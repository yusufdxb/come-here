# Latency Optimization: Implementation Tasks

**Target:** <= 0.3s from phrase-end to rotation command publish
**Condition:** Target is contingent on Phase 0 benchmark results
**Date prepared:** 2026-04-07
**Full plan:** ~/Documents/Obsidian Vault/Projects/come-here/latency-optimization-plan.md

---

## Phase 0: Benchmark (MUST complete first)

Run these on Jetson before any refactor work. Results determine backend choice.

### Task 0.1: Benchmark Whisper inference

**Script:** `come_here_audio/scripts/benchmark_whisper.py` (already written)

```bash
# On Jetson:
cd ~/come-here
python3 -u come_here_audio/scripts/benchmark_whisper.py \
    --models tiny.en base.en \
    --devices cuda cpu \
    --windows 1.0 1.2 \
    --n-runs 50
```

**Collect:** p95 inference time for each config. Key number: `tiny.en / cuda / float16 / 1.0s window`.

**Gates:**
- Gate A: tiny.en CUDA p95 <= 100ms -> proceed with faster-whisper backend
- Gate B: tiny.en CUDA p95 > 150ms -> switch plan to whisper_trt
- Gate C: 100-150ms -> relax target to 0.4s, plan whisper_trt migration as follow-up

### Task 0.2: Benchmark DOA polling

**Script:** `come_here_audio/scripts/benchmark_doa_poll.py` (already written)

```bash
python3 -u come_here_audio/scripts/benchmark_doa_poll.py
```

**Collect:** USB HID round-trip p95, max sustainable polling rate, DOA data freshness.
**Use result to set:** `poll_rate_hz` parameter in continuous DOA polling (Task 2.2).

### Task 0.3: Measure current demo baseline

Add timestamps to existing `hear_and_rotate_demo.py` (don't rewrite yet):

```python
import time
# Before sd.rec:
t_rec_start = time.monotonic()
# After sd.wait():
t_rec_end = time.monotonic()
# Before model.transcribe():
t_infer_start = time.monotonic()
# After transcribe loop:
t_infer_end = time.monotonic()
# Before play_wav_on_robot:
t_play_start = time.monotonic()
# Before sport_pub.publish(rotate):
t_rotate_pub = time.monotonic()
# Log:
print(f"[LATENCY] rec={t_rec_end-t_rec_start:.3f}s infer={t_infer_end-t_infer_start:.3f}s "
      f"play={t_rotate_pub-t_play_start:.3f}s total={t_rotate_pub-t_rec_start:.3f}s")
```

Run 10 trials, report baseline e2e latency.

### Task 0.4: Download tiny.en model to Jetson

```bash
# On PC with internet:
pip download faster-whisper  # if needed
# Or use python to cache:
python3 -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en')"
# Copy to Jetson:
scp -r ~/.cache/huggingface/hub/models--Systran--faster-whisper-tiny.en \
    unitree@172.20.10.6:~/come-here/models/faster-whisper-tiny.en
```

---

## Phase 1: Demo Path Optimization (after Phase 0 gates pass)

Optimize `hear_and_rotate_demo.py` first -- this is the hardware path that matters.
Shared code (detector, DOA provider) benefits both demo and ROS paths.

### Task 1.1: Create ring_buffer.py

**Status:** Already written at `come_here_audio/come_here_audio/ring_buffer.py`
**Contains:** `RingBuffer` (numpy circular buffer, thread-safe) + `LatestOnlyQueue` (single-slot drop-old queue)
**Test offline:** Unit tests with synthetic writes/reads, verify wraparound and thread safety.

### Task 1.2: Rewrite whisper_phrase_detector.py with streaming capture

**File:** `come_here_audio/come_here_audio/whisper_phrase_detector.py`

Replace the blocking `_listen_loop` with three components:

1. **InputStream callback** (`_audio_callback`):
   - `sounddevice.InputStream`, blocksize ~1600 (100ms at 16kHz)
   - Extract beam channel (ch1), apply gain (4x), write mono to RingBuffer
   - Runs in PortAudio's thread -- keep it fast, no allocations

2. **Segmenter thread** (`_segmenter_loop`):
   - Sleep for hop interval (~250ms), read last window (~1.0s) from RingBuffer
   - Energy-based VAD: `np.mean(window ** 2)` vs threshold
   - State machine: SILENCE -> SPEECH -> TRAILING_SILENCE
   - On trailing silence >= ~200ms: extract window, put into LatestOnlyQueue
   - Record `t_speech_end = time.monotonic()` at endpoint

3. **Inference thread** (`_inference_loop`):
   - Block on `LatestOnlyQueue.get()` -- stale segments auto-dropped
   - Run `model.transcribe()` with `beam_size=1, language="en", initial_prompt="come here"`
   - On match: put `PhraseDetection` into `self._detections` (backward compat)
   - Fire `self._on_detection_cb(detection, t_speech_end)` if registered

**New public method:** `set_on_detection(callback)` for event-driven notification.

**New constructor params:**
- `mic_gain: float = 4.0`
- `window_duration_s: float = 1.0`
- `hop_duration_ms: int = 250`
- `end_silence_ms: int = 200`
- `ring_buffer_duration_s: float = 3.0`
- `energy_threshold: float = 0.001`

**Backward compat:** `check()` still works (reads from `_detections` queue).

**Concrete values are initial estimates.** Tune `window_duration_s`, `hop_duration_ms`, `energy_threshold` on hardware.

### Task 1.3: Add continuous DOA polling to ReSpeakerDOAProvider

**File:** `come_here_audio/come_here_audio/respeaker_doa_provider.py`

Add:
- `start_continuous(poll_rate_hz: float)` -- starts daemon thread polling at measured rate
- Thread stores `collections.deque(maxlen=N)` of `(time.monotonic(), azimuth_rad, vad_active: bool)`
- `get_latched_direction(window_s: float = 1.0) -> Optional[DirectionEstimate]`
  - Returns median azimuth of VAD-active samples within last `window_s` seconds
  - Falls back to most recent VAD-active sample if < 3 samples
  - Returns `None` if no VAD-active samples in window
- Keep existing `get_direction()` unchanged
- `teardown()` stops polling thread

**Poll rate** is a constructor param, set from Task 0.2 benchmark result. Default 30 Hz as safe initial.

### Task 1.4: Rewrite hear_and_rotate_demo.py

**File:** `come_here_audio/scripts/hear_and_rotate_demo.py`

**Changes:**
- Replace inline `whisper_thread` with reusable `WhisperPhraseDetector` instance
- Use `set_on_detection()` callback instead of polling `detection_q`
- In callback: latch DOA via `doa.get_latched_direction()`, immediately publish rotation
- Move `play_wav_on_robot()` to a fire-and-forget daemon thread (off critical path)
- Backend config from Phase 0: `device="cuda"`, model size from gate decision
- Fix gain to 4.0 (currently 10.0 which distorts)

**Critical path after rewrite:**
```
phrase end -> segmenter detects endpoint (~200ms) -> inference (~TBD from benchmark) -> callback -> sport_pub.publish
```

No playback blocking. No timer hops. DOA latched from continuous polling.

### Task 1.5: Add latency instrumentation to demo

Record `time.monotonic()` at:
- `t_speech_end`: segmenter endpoint detection
- `t_infer_start` / `t_infer_done`: around transcribe()
- `t_match`: phrase match
- `t_doa_latch`: after get_latched_direction()
- `t_rotate_pub`: after sport_pub.publish()

Log: `[LATENCY] speech_end->rotate_pub={t_rotate_pub - t_speech_end:.4f}s infer={t_infer_done - t_infer_start:.4f}s`

Run 10+ trials, report p95 of `t_rotate_pub - t_speech_end`.

---

## Phase 2: ROS Node Path (after demo path validated)

Port optimizations to the ROS node graph. Shared code from Phase 1 is reused.

### Task 2.1: Extend WakePhrase.msg with latched direction

**File:** `come_here_msgs/msg/WakePhrase.msg`

```
# Wake phrase detection event with latched sound direction
std_msgs/Header header
string phrase
float64 confidence
float64 azimuth_rad           # latched DOA at time of utterance
float64 direction_confidence   # DOA confidence (0.0 if no VAD samples)
```

Requires `colcon build` to regenerate Python bindings.

### Task 2.2: Update audio_node.py for event-driven wake

**File:** `come_here_audio/come_here_audio/audio_node.py`

- Switch imports from `std_msgs` to `come_here_msgs.msg` (AudioDirection, WakePhrase)
- Remove wake phrase polling from `_tick` timer
- Register `set_on_detection()` callback on WhisperPhraseDetector
- In callback: call `doa.get_latched_direction()`, populate extended WakePhrase, publish immediately
- Keep AudioDirection publishing at 10Hz in `_tick` for monitoring (behavior_node no longer depends on it for wake-to-rotate)
- Pass streaming params to WhisperPhraseDetector constructor

### Task 2.3: Update behavior_node.py -- remove LISTENING state

**File:** `come_here_behavior/come_here_behavior/behavior_node.py`

- Switch imports from `std_msgs` to `come_here_msgs.msg` (WakePhrase, PersonDetection)
- Subscribe to `/come_here/wake_phrase` with type `WakePhrase` (extended)
- `_wake_cb` reads `azimuth_rad` and `direction_confidence` from message
- If `direction_confidence >= threshold`: transition IDLE -> TURN_TO_SOUND directly, publish rotation immediately in callback
- If `direction_confidence < threshold`: log warning, stay IDLE (no stale DOA race)
- Remove LISTENING state from enum and `_tick` logic
- Remove dependency on separate `/come_here/audio_direction` for wake-to-rotate
- Keep 10Hz timer for SEARCH_FOR_PERSON and APPROACH_PERSON

### Task 2.4: Launch "I am coming" off critical path (ROS node)

Add acknowledgment playback as a fire-and-forget action triggered from `_wake_cb` in behavior_node (or a separate lightweight node). Must not block rotation.

---

## Summary: Must-Do vs Nice-to-Have

### Must-do for latency target
- [x] Benchmark scripts (Phase 0) -- prepared
- [x] ring_buffer.py -- written
- [ ] Streaming capture (InputStream + RingBuffer)
- [ ] Overlapping capture/inference (segmenter + inference threads)
- [ ] LatestOnlyQueue (drop stale segments)
- [ ] Latched DOA on wake event (continuous polling)
- [ ] Event-driven wake-to-rotate (no timer hops)
- [ ] Acknowledgment playback off critical path
- [ ] WakePhrase.msg extension with direction fields

### Nice-to-have (after must-do validated)
- [ ] `initial_prompt="come here"` in Whisper (trivial, +reliability)
- [ ] TensorRT FP16 export for YOLO11n (~2x vision speed)
- [ ] Silero VAD (if onnxruntime works on ARM, otherwise energy VAD is sufficient)
- [ ] whisper_trt backend (only if Gate B triggers)

---

## Files Already Prepared (on local clone ~/workspace/come-here/)

| File | Status | Notes |
|------|--------|-------|
| `come_here_audio/scripts/benchmark_whisper.py` | New, ready | Run on Jetson for Phase 0 |
| `come_here_audio/scripts/benchmark_doa_poll.py` | New, ready | Run on Jetson for Phase 0 |
| `come_here_audio/come_here_audio/ring_buffer.py` | New, ready | RingBuffer + LatestOnlyQueue, needs unit tests |
| All other files | Unmodified | Changes described above, not yet applied |

---

## Execution Order

```
0.1  benchmark_whisper.py on Jetson       <- FIRST, determines backend
0.2  benchmark_doa_poll.py on Jetson      <- sets DOA polling rate
0.3  Instrument current demo baseline     <- establishes "before" number
0.4  Download tiny.en to Jetson           <- needed for benchmarks

-- GATE DECISION --

1.1  ring_buffer.py unit tests            <- already written, just test
1.2  Rewrite whisper_phrase_detector.py   <- biggest change
1.3  Add continuous DOA to respeaker      <- medium change
1.4  Rewrite hear_and_rotate_demo.py      <- wire it all together
1.5  Measure optimized demo latency       <- validates target

-- DEMO PATH VALIDATED --

2.1  Extend WakePhrase.msg + colcon build <- message migration
2.2  Update audio_node.py                 <- event-driven wake
2.3  Update behavior_node.py              <- remove LISTENING state
2.4  Async acknowledgment in ROS path     <- off critical path
```
