# Codebase Concerns

**Analysis Date:** 2026-04-06

## Tech Debt

**Incomplete Hardware Integration:**
- Issue: All three nodes (audio, perception, behavior) use mock implementations that will need to be replaced with real hardware drivers
- Files: `come_here_audio/come_here_audio/audio_node.py`, `come_here_perception/come_here_perception/perception_node.py`, `come_here_behavior/come_here_behavior/behavior_node.py`
- Impact: System cannot run on actual GO2 hardware until real implementations are added. Currently crashes with `NotImplementedError` if `use_mock:=false`
- Fix approach: Implement actual `AudioDirectionProvider` for microphone array, `PersonDetector` for camera-based vision, and GO2 motion commands via `cmd_rotate`/`cmd_move` publishers or direct SDK calls

**Message Type Placeholders:**
- Issue: Audio and behavior nodes use `std_msgs/Float64MultiArray` and `std_msgs/String` instead of custom typed messages from `come_here_msgs`
- Files: `come_here_audio/come_here_audio/audio_node.py` (lines 23-25, 79-85), payload unpacking in `come_here_behavior/come_here_behavior/behavior_node.py` (lines 94-104)
- Impact: Type safety is lost; array indices are magic numbers `[0]`, `[1]`, `[2]`, `[3]`. Breaks if message order changes. Hard to validate data at boundaries
- Fix approach: Build `come_here_msgs` package (appears to be created but not built), define custom `AudioDirection.msg`, `WakePhrase.msg`, `PersonDetection.msg` messages, update all publishers/subscribers

**Bare Exception Handling in Critical Thread:**
- Issue: `WhisperPhraseDetector._listen_loop()` catches all `Exception` types with generic handler that only logs via `time.sleep()`
- Files: `come_here_audio/come_here_audio/whisper_phrase_detector.py` (lines 194-198)
- Impact: Silent failures in audio thread; non-critical errors (e.g., missing sounddevice module) don't propagate, node appears healthy but stops detecting. Operator unaware of failures
- Fix approach: Log specific exception types with details, raise for unrecoverable errors (ImportError, device initialization), only catch transient audio capture errors

**Silent Failures in Transcription Quality Check:**
- Issue: `WhisperPhraseDetector` silently skips silent audio chunks and low-confidence transcriptions without feedback
- Files: `come_here_audio/come_here_audio/whisper_phrase_detector.py` (lines 186-187, 214-215)
- Impact: Detection failures are invisible; difficult to diagnose if user hasn't heard the wake phrase or audio hardware failed. No metrics exposed
- Fix approach: Add debug logging with audio levels and confidence scores; expose ROS topics for silence percentage and average confidence

---

## Known Bugs

**Thread Join Timeout Race Condition:**
- Symptoms: Audio node may hang for 5+ seconds on shutdown if Whisper inference is slow
- Files: `come_here_audio/come_here_audio/whisper_phrase_detector.py` (lines 158-159)
- Trigger: Shutdown while `_listen_loop()` is in `sd.rec()` or Whisper `transcribe()` blocking call
- Workaround: Use `timeout` parameter in `thread.join(5.0)`, but this leaves thread running in background consuming resources

**Floating-Point Audio Silence Detection Too Strict:**
- Symptoms: Actual speech recorded at low volume (RMS < 0.01) is skipped entirely, never transcribed
- Files: `come_here_audio/come_here_audio/whisper_phrase_detector.py` (line 186)
- Trigger: Low-gain microphone or user speaking quietly
- Workaround: User must speak louder; no adaptive gain control

**State Machine Doesn't Validate Person Detection Data:**
- Symptoms: If perception node sends invalid data (NaN, Inf, missing fields), behavior state machine reads garbage values
- Files: `come_here_behavior/come_here_behavior/behavior_node.py` (lines 99-104)
- Trigger: Perception node crashes and publishes default values, or msg format changes
- Workaround: Manual inspection of `/come_here/person_detection` topic with `ros2 topic echo`

---

## Security Considerations

**No Authentication for Mock Control Topics:**
- Risk: Any process can publish to `/come_here/mock_trigger` and `/come_here/mock_person`, spoofing sensor input
- Files: `come_here_audio/come_here_audio/audio_node.py` (lines 88-92), `come_here_perception/come_here_perception/perception_node.py` (lines 42-45)
- Current mitigation: Mock mode only active in development; disabled via `use_mock: false` parameter
- Recommendations: In production, remove mock topic subscriptions entirely; if needed for debugging, require authentication token in message payload

**Model Path Traversal Risk in Whisper Adapter Loading:**
- Risk: `whisper_adapter_path` parameter accepts arbitrary file paths; user could load malicious LoRA adapters
- Files: `come_here_audio/come_here_audio/audio_node.py` (lines 62-68), `come_here_audio/come_here_audio/whisper_phrase_detector.py` (lines 144)
- Current mitigation: Parameter loading is local-only (no remote config), must be set at launch time
- Recommendations: Validate adapter path is within expected directory; sign/verify adapters with checksums

**No Validation of Confidence Thresholds:**
- Risk: Negative or >1.0 confidence values accepted by state machine, causing unpredictable behavior
- Files: `come_here_behavior/come_here_behavior/behavior_node.py` (lines 44-54 parameter declarations, lines 119, 137 threshold comparisons)
- Current mitigation: None; relies on config file integrity
- Recommendations: Add bounds checking in `__init__` to validate `0.0 <= threshold <= 1.0`

---

## Performance Bottlenecks

**Whisper Inference Blocks Audio Recording:**
- Problem: `_listen_loop()` records a 2-second chunk, then blocks for 1-5 seconds during Whisper inference before recording next chunk. Any wake phrase detection is delayed by full chunk duration + inference time
- Files: `come_here_audio/come_here_audio/whisper_phrase_detector.py` (lines 170-192)
- Cause: Single-threaded recording+inference; sounddevice `rec()` is synchronous, Whisper inference is synchronous
- Improvement path: Record audio in one thread to a ring buffer, inference in separate thread from buffer; reduces latency to near-realtime

**No Throttling on Behavior State Transitions:**
- Problem: At 10 Hz tick rate, state machine publishes to `/come_here/cmd_rotate` and `/come_here/cmd_move` at 10 Hz even when values haven't changed; GO2 hardware is overloaded with redundant commands
- Files: `come_here_behavior/come_here_behavior/behavior_node.py` (lines 125-127, 158-164)
- Cause: No check for state/command changes before publishing
- Improvement path: Only publish when command value changes, or batch multiple ticks worth of stable commands

**HuggingFace Model Loading Not Cached:**
- Problem: Fine-tuned Whisper model loaded fresh on every node startup, taking 10+ seconds
- Files: `come_here_audio/come_here_audio/whisper_phrase_detector.py` (lines 130-148)
- Cause: `WhisperForConditionalGeneration.from_pretrained()` downloads/loads model weights fresh each time
- Improvement path: Cache loaded model in class variable or use HF cache directory (already done via HF defaults, but verify `~/.cache/huggingface` is persistent)

---

## Fragile Areas

**State Machine Time Tracking:**
- Files: `come_here_behavior/come_here_behavior/behavior_node.py` (lines 64, 133, 136, 147)
- Why fragile: Uses `self.get_clock().now()` to track search timeout, but clock can jump (NTP sync, sim time in tests). `_search_start_time` is set to `None` initially and only populated on state transition; accessing it before transition causes `TypeError`
- Safe modification: Add strict type hint for `_search_start_time: Optional[Time]`, validate before arithmetic (`if self._search_start_time is None: return`)
- Test coverage: No test for SEARCH_FOR_PERSON timeout logic; `test_state_machine.py` only checks state enum existence

**Message Array Unpacking with No Length Validation:**
- Files: `come_here_behavior/come_here_behavior/behavior_node.py` (lines 94-104)
- Why fragile: Checks `if len(msg.data) >= 2` for direction, but no assertion for exact length. If perception node sends wrong number of fields, rest silently ignored
- Safe modification: Define exactly-sized message types in `come_here_msgs`; use dataclass unpacking instead of array indices
- Test coverage: No integration test for message format changes

**Confidence Threshold Comparison Without Bounds:**
- Files: `come_here_behavior/come_here_behavior/behavior_node.py` (lines 44-54, 119, 137)
- Why fragile: Parameters declared but never validated; if set to -1.0 or 2.0, comparisons become nonsensical
- Safe modification: Add validation in `__init__`: `assert 0.0 <= self._dir_threshold <= 1.0`
- Test coverage: No test for invalid parameter values

**Whisper Model Device Dtype Mismatch Risk:**
- Files: `come_here_audio/come_here_audio/whisper_phrase_detector.py` (lines 135-147)
- Why fragile: CPU model uses `float32`, GPU uses `float16`. If adapter is trained on GPU but node runs on CPU (or vice versa), dtype mismatch causes inference failure
- Safe modification: Always use consistent dtype; detect device and set dtype before loading model, document requirement
- Test coverage: No test for GPU/CPU dtype handling

---

## Scaling Limits

**Single Whisper Model Instance Per Node:**
- Current capacity: ~1 transcription every 2 seconds (chunk_duration_s) + inference time (~1-3 sec)
- Limit: If multiple simultaneous audio sources needed (e.g., array of mics), single model cannot handle it
- Scaling path: Create model server (separate process) shared by multiple audio nodes, or horizontally scale with separate Whisper instances per audio source

**Queue.Queue Unbounded Detection History:**
- Current capacity: All detections ever queued in memory
- Limit: If Whisper detects extremely frequently, queue grows without bound; high memory usage
- Scaling path: Use `queue.Queue(maxsize=N)` to drop oldest detections on overflow; or move to ROS native message queue

**ROS Topic Publishing at Fixed 10 Hz:**
- Current capacity: All three nodes publish at 10 Hz regardless of message rate or system load
- Limit: Jetson Orin NX CPU budget is limited; 30 msg/sec might be acceptable but not benchmarked
- Scaling path: Measure actual CPU usage; adjust publish rates dynamically or use ROS2 QoS settings to reduce message flow

---

## Dependencies at Risk

**Faster-Whisper Not Tested on Jetson:**
- Risk: `faster-whisper` (CTranslate2 backend) targets x86 SIMD optimizations; Jetson Orin NX uses ARM NEON. Performance unverified
- Impact: May be 5-10x slower than expected; inference time balloons from 1 sec to 10+ sec
- Migration plan: Benchmark on actual hardware; if too slow, use HuggingFace transformers backend instead (supports ARM), accept higher latency, or use quantized model (int8)

**SoundDevice Module Dependency:**
- Risk: `sounddevice` may not be packaged for ROS 2 Humble; requires native audio backend (ALSA, PulseAudio, Jack)
- Impact: Hard to debug audio initialization failures; microphone not found errors
- Migration plan: Test on Jetson Orin NX; if audio stack missing, install `alsa-utils` and `libasound2`; consider using PyAudio alternative

**PyTorch Availability on Jetson:**
- Risk: Fine-tuned model evaluation requires torch + transformers + peft; total ~2GB download
- Impact: Slow first-time setup; storage constraints on Jetson (internal eMMC ~32GB)
- Migration plan: Pre-cache model to T7 external drive; document offline setup procedure

---

## Missing Critical Features

**No Recovery Mechanism for Lost Person:**
- Problem: If person detection fails (camera occlusion, lighting), robot transitions to SEARCH_FOR_PERSON but has no re-acquisition strategy
- Blocks: Multi-attempt approach logic; robot gives up after 10 seconds
- Recommendation: Add BACKTRACK state to return to last known position, or SEARCH_EXPAND to scan wider area

**No Feedback to User When Robot Arrives:**
- Problem: Behavior node transitions to STOP silently; user doesn't know if robot reached the source or gave up
- Blocks: Closure confirmation; user must monitor ROS topics
- Recommendation: Add audio feedback (beep or TTS) or LED signal when STOP state entered

**No Timeout or Fallback If Audio Never Detected:**
- Problem: If wake phrase never detected, robot stays in IDLE forever
- Blocks: Time-based activation (e.g., scheduled wake-ups)
- Recommendation: Add optional `startup_timeout_s` parameter; transition to SEARCH_FOR_PERSON automatically after timeout

**No Graceful Degradation for Partial Sensor Failure:**
- Problem: If perception node crashes while behavior node runs, behavior uses stale person_detected=False forever
- Blocks: Fault tolerance; system has no awareness of sensor health
- Recommendation: Add heartbeat subscription; transition to SEARCH_FOR_PERSON if perception hasn't published in >2 seconds

---

## Test Coverage Gaps

**No Integration Tests for Node Communication:**
- What's not tested: Message flow between audio → behavior → perception nodes; no launch-time end-to-end test
- Files: `come_here_audio/test/test_audio_provider.py`, `come_here_perception/test/test_person_detector.py`, `come_here_behavior/test/test_state_machine.py` only test isolated modules
- Risk: Breaking changes in message schema go undetected until runtime on hardware
- Priority: **HIGH** - Add fixture-based ROS2 node tests with pytest-ros

**No Whisper Inference Tests:**
- What's not tested: `WhisperPhraseDetector` threading, queue behavior, actual audio processing
- Files: No test file for `whisper_phrase_detector.py` or `finetune_whisper.py`
- Risk: Silent failure in background thread undetected; confidence calculation bugs, adapter loading failures
- Priority: **HIGH** - Mock sounddevice input, test queue output, verify confidence scaling

**No Behavior State Machine Transition Coverage:**
- What's not tested: State transitions in sequence (IDLE → LISTENING → TURN_TO_SOUND → SEARCH_FOR_PERSON → APPROACH_PERSON → STOP). Only state enum tested
- Files: `come_here_behavior/test/test_state_machine.py` (3 lines)
- Risk: Logic errors in timeout, distance checks, person loss recovery undetected
- Priority: **HIGH** - Add parametrized state transition tests with mock clock and message injection

**No Training/Evaluation Tests for Model Quality:**
- What's not tested: Fine-tuning loop doesn't validate output on known test set automatically; evaluation script manual-only
- Files: `training/finetune_whisper.py`, `training/evaluate.py` (standalone scripts, no automated test)
- Risk: Poor-quality adapters deployed without metrics awareness
- Priority: **MEDIUM** - Add acceptance test threshold (e.g., min 90% F1 score required)

**No Parameter Validation Tests:**
- What's not tested: Behavior node parameters (negative speeds, NaN thresholds, >1.0 confidence)
- Files: No test for parameter bounds in `come_here_behavior/come_here_behavior/behavior_node.py`
- Risk: Config errors crash node at runtime
- Priority: **MEDIUM** - Parametrize test with invalid values, verify rejection or safe defaults

**No Microphone Hardware Fallback Tests:**
- What's not tested: Audio node with missing sounddevice module, unavailable audio device, permission errors
- Files: No test for exception paths in `audio_node.py`, `whisper_phrase_detector.py`
- Risk: Deployment on Jetson fails silently if audio stack not configured
- Priority: **MEDIUM** - Mock sounddevice import failures, verify graceful error messages

---

*Concerns audit: 2026-04-06*
