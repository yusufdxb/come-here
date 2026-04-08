# Latency Optimization — Jetson Execution Prompt

Paste everything below the line into Claude Code when connected to the Jetson.

---

You are resuming a pre-planned latency optimization for the `come-here` ROS 2 project. The full plan, benchmark scripts, and shared infrastructure were prepared offline. Your job now is to **execute the plan on this Jetson**.

Read these files first to load full context:
1. `LATENCY_IMPLEMENTATION_TASKS.md` — the step-by-step execution checklist
2. `CLAUDE.md` — project architecture and conventions

## What was already prepared (do NOT rewrite these)

| File | What it is |
|------|-----------|
| `come_here_audio/scripts/benchmark_whisper.py` | Phase 0: Whisper inference benchmark (tiny.en/base.en, CUDA/CPU, 1.0s/1.2s, p95 gates) |
| `come_here_audio/scripts/benchmark_doa_poll.py` | Phase 0: USB HID DOA polling round-trip and rate sweep |
| `come_here_audio/come_here_audio/ring_buffer.py` | RingBuffer (numpy circular, thread-safe) + LatestOnlyQueue (single-slot, drop-old) |

## What you must do now — in this exact order

### PHASE 0: BENCHMARKS (mandatory, determines everything else)

**Step 1:** Ensure the tiny.en model is available. Check if `models/faster-whisper-tiny.en` exists. If not, download it:
```python
python3 -c "from faster_whisper import WhisperModel; m = WhisperModel('tiny.en', device='cpu', compute_type='int8'); print('tiny.en cached')"
```
If that fails (no internet on Jetson), check if it can be found in pip cache or HF cache. If truly missing, STOP and tell the user to scp it from the PC.

**Step 2:** Run the Whisper benchmark:
```bash
python3 -u come_here_audio/scripts/benchmark_whisper.py --models tiny.en base.en --devices cuda cpu --windows 1.0 1.2 --n-runs 50
```
**Read the output.** Extract the p95 for `tiny.en / cuda / float16 / 1.0s`. This is the gate number.

**Step 3:** Make the gate decision. This is NON-NEGOTIABLE — do not skip or assume:
- **Gate A** (p95 <= 100ms): Proceed with faster-whisper tiny.en CUDA as the backend. Use `device="cuda", compute_type="float16", model_size="tiny.en"` everywhere.
- **Gate B** (p95 > 150ms): STOP the refactor. Tell the user: "tiny.en CUDA is too slow (p95={X}ms). The plan calls for whisper_trt via jetson-containers. Set that up first, then resume." Do not proceed with Phase 1.
- **Gate C** (100ms < p95 <= 150ms): Proceed with faster-whisper but adjust the target to 0.4s. Note this in every latency report. Flag that whisper_trt should be a follow-up.

**Step 4:** Run the DOA benchmark (only if ReSpeaker is plugged in):
```bash
python3 -u come_here_audio/scripts/benchmark_doa_poll.py
```
Record the recommended polling rate. Use it as `poll_rate_hz` in Task 1.3. If ReSpeaker is not connected, default to 30 Hz and move on.

**Step 5:** Instrument the current demo to measure baseline latency. Add `time.monotonic()` timestamps to `hear_and_rotate_demo.py` at: before sd.rec, after sd.wait, before/after transcribe, before play_wav, before sport_pub.publish. Log them. Run the demo 5+ times and report the baseline phrase-end-to-rotation time. Then revert the instrumentation (the demo will be fully rewritten in Phase 1).

### PHASE 1: DEMO PATH OPTIMIZATION (only after Phase 0 gates pass)

Execute these tasks from `LATENCY_IMPLEMENTATION_TASKS.md` in order. Use agents to parallelize where dependencies allow.

**Task 1.1:** Run ring_buffer.py through quick unit tests (write a test, run it, confirm RingBuffer and LatestOnlyQueue work).

**Task 1.2:** Rewrite `whisper_phrase_detector.py` with streaming capture. This is the biggest change. Follow the spec in LATENCY_IMPLEMENTATION_TASKS.md exactly:
- InputStream callback -> RingBuffer -> Segmenter thread (energy VAD) -> LatestOnlyQueue -> Inference thread
- Add `set_on_detection(callback)` for event-driven notification
- Add `mic_gain`, `window_duration_s`, `hop_duration_ms`, `end_silence_ms`, `ring_buffer_duration_s`, `energy_threshold` params
- Keep `check()` working for backward compat
- Use the backend config from your Phase 0 gate decision
- Add `initial_prompt="come here"` to the transcribe call

**Task 1.3:** Add continuous DOA polling to `respeaker_doa_provider.py`:
- `start_continuous(poll_rate_hz)` — daemon thread, timestamped deque
- `get_latched_direction(window_s)` — median of VAD-active samples in window
- Use the poll rate from Phase 0 Step 4

**Task 1.4:** Rewrite `hear_and_rotate_demo.py`:
- Use the rewritten WhisperPhraseDetector (don't duplicate logic)
- Use `set_on_detection()` callback
- In callback: latch DOA, fire-and-forget "I am coming" in daemon thread, publish rotation IMMEDIATELY
- Fix gain to 4.0 (not 10.0)
- Use CUDA backend from gate decision

**Task 1.5:** Add latency instrumentation and test:
- Record timestamps at speech_end, infer_start, infer_done, match, doa_latch, rotate_pub
- Log `[LATENCY] speech_end->rotate_pub=Xs infer=Xs`
- Run the demo 10+ times, report p95 of speech_end to rotate_pub
- Compare against Phase 0 baseline

**Report Phase 1 results before proceeding to Phase 2.**

### PHASE 2: ROS NODE PATH (only after demo path validates the target)

**Task 2.1:** Extend `come_here_msgs/msg/WakePhrase.msg`:
```
std_msgs/Header header
string phrase
float64 confidence
float64 azimuth_rad
float64 direction_confidence
```
Run `colcon build --packages-select come_here_msgs` to regenerate bindings.

**Task 2.2:** Update `audio_node.py`:
- Import from `come_here_msgs.msg` instead of `std_msgs`
- Remove wake polling from `_tick`
- Register `set_on_detection()` callback, publish extended WakePhrase immediately with latched DOA
- Pass streaming params to WhisperPhraseDetector

**Task 2.3:** Update `behavior_node.py`:
- Import from `come_here_msgs.msg`
- `_wake_cb` reads azimuth from extended WakePhrase, transitions IDLE -> TURN_TO_SOUND directly
- Remove LISTENING state entirely
- Publish rotation in callback, not deferred to tick
- Fire "I am coming" in daemon thread

**Task 2.4:** Build and test:
```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch come_here_bringup come_here.launch.py use_mock:=false
```

## Hard constraints during execution

1. **Phase 0 is mandatory.** Do not skip benchmarks. Do not assume timing. Use p95, not mean.
2. **If Gate B triggers, STOP.** Tell the user to set up whisper_trt. Do not proceed with a slow backend.
3. **Demo path first.** Do not touch ROS nodes (audio_node, behavior_node) until the demo validates the latency target.
4. **Use proper messages.** Extend WakePhrase.msg with direction fields. Do NOT use Float64MultiArray for wake events.
5. **All latency numbers are conditional until measured.** Label every timing as verified/inferred/hardware-unverified.
6. **Do not break backward compat.** `check()` must still work on WhisperPhraseDetector. Existing `get_direction()` must still work on DOA provider.
7. **Commit after each phase completes successfully.** Phase 0 results, Phase 1 demo rewrite, Phase 2 ROS migration — three separate commits.

## After completion

Report a summary with:
- Phase 0 benchmark results (table of p95 values, gate decision)
- Phase 1 latency results (p95 speech_end to rotate_pub, before vs after)
- Phase 2 status (built, tested, or skipped)
- Any open issues or tuning needed
