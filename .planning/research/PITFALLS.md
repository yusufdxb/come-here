# Domain Pitfalls

**Domain:** Robot voice-commanded approach system (ReSpeaker + Whisper + YOLO + GO2 locomotion on Jetson Orin NX)
**Researched:** 2026-04-06

---

## Critical Pitfalls

Mistakes that cause hardware damage, system rewrites, or weeks of debugging.

### Pitfall 1: Sport Mode Collision Destroys Actuators

**What goes wrong:** Publishing velocity or low-level joint commands while GO2 Sport Mode is still active causes loud mechanical noises and unpredictable jerking. Sport Mode runs on the MCU and injects commands directly into DDS -- it is not a ROS 2 node you can kill. Your commands and Sport Mode's commands fight over the same actuators simultaneously.

**Why it happens:** Developers assume they can just publish to `/cmd_vel` or `/lowcmd` and the robot will respond. Sport Mode is always-on at boot and intercepts all motion commands at the DDS layer.

**Consequences:** Motor damage, gearbox stress, robot falls unpredictably. In the worst case, the robot collapses if Sport Mode is released without first laying the robot down (motors de-energize instantly).

**Prevention:**
1. Use `MotionSwitcherClient.ReleaseMode()` to disable Sport Mode before sending any custom motion commands
2. Always call `StandDown()` before releasing Sport Mode so the robot is already on the ground when motors de-energize
3. After releasing Sport Mode, re-stand the robot under your own low-level control
4. For high-level approach behavior, use the Sport Mode API (`/api/sport/request` with `SportClient`) instead of fighting it -- send velocity commands through the sport service, not raw `/cmd_vel`

**Detection:** Robot makes grinding/clicking noises on first command. Unexpected jerky motion. Robot collapses when you thought it would walk.

**Phase relevance:** Locomotion bridge phase. Must be the first thing validated before any approach behavior is tested.

**Sources:**
- [How to disable sport mode on the Go2](https://ric.engineering/posts/Unitree-Sportmode/)
- [Sport Mode Control - DeepWiki](https://deepwiki.com/unitreerobotics/unitree_ros2/4.2-sport-mode-control)

---

### Pitfall 2: CTranslate2 (faster-whisper) Fails to Load on Jetson ARM

**What goes wrong:** `faster-whisper` depends on CTranslate2, which ships pip wheels compiled for x86 only. On Jetson Orin NX (ARM/aarch64), `import ctranslate2` fails with `ImportError: libctranslate2.so.4: cannot open shared object file`. The pip package installs Python bindings but not the C++ shared library needed on ARM.

**Why it happens:** CTranslate2's pip distribution targets x86 with SIMD optimizations. There are no official ARM wheels. Installing via pip appears to succeed but the native library is missing.

**Consequences:** Whisper inference completely non-functional on Jetson. Days lost debugging import errors. Fallback to PyTorch Whisper uses 3-5x more memory and is significantly slower.

**Prevention:**
1. Build CTranslate2 from source using the [jetson-containers build script](https://github.com/dusty-nv/jetson-containers/blob/master/packages/ml/ctranslate2/build.sh) -- this compiles with CUDA and cuDNN support for ARM
2. Alternatively, use NVIDIA's [whisper_trt](https://github.com/NVIDIA-AI-IOT/whisper_trt) which runs ~3x faster and uses ~60% memory vs PyTorch on Jetson Orin
3. Test the import on Jetson early -- do not assume pip install success means runtime success
4. Have a fallback plan: HuggingFace transformers backend with `float16` on GPU works on ARM without custom builds

**Detection:** `import ctranslate2` throws ImportError immediately. Easy to catch in a smoke test.

**Phase relevance:** Audio/Whisper integration phase. Must be validated on Jetson hardware before any Whisper work proceeds. This is a blocker that should be tested in the first hours of the phase.

**Sources:**
- [CTranslate2 issue on Jetson - NVIDIA Forums](https://forums.developer.nvidia.com/t/ctranslate-2-issue/335263)
- [whisper_trt - NVIDIA](https://github.com/NVIDIA-AI-IOT/whisper_trt)

---

### Pitfall 3: GPU Memory Exhaustion Running YOLO + Whisper Concurrently

**What goes wrong:** Jetson Orin NX has 8-16 GB unified memory shared between CPU and GPU. Loading both a YOLO model (TensorRT engine) and a Whisper model (PyTorch or CTranslate2) simultaneously can exceed available GPU memory, causing CUDA OOM errors or system-wide memory pressure that triggers the Linux OOM killer.

**Why it happens:** Jetson's unified memory architecture means CPU allocations compete with GPU allocations. YOLO TensorRT engine workspace + weights (~500MB-2GB depending on model size and precision) plus Whisper model (~500MB-1.5GB) plus CUDA context overhead (~300MB per process) can approach or exceed total memory. Two separate processes each create their own CUDA context, doubling overhead.

**Consequences:** Random CUDA OOM crashes during inference. System becomes unresponsive. Nodes killed by OOM killer with no recovery. If running in separate processes, each CUDA context wastes ~300MB.

**Prevention:**
1. Use YOLOv8n or YOLOv11n (nano) with TensorRT FP16 or INT8 -- keeps engine under 50MB with workspace under 1GB
2. Use Whisper tiny or base model, not small/medium -- base.en is ~150MB in CTranslate2
3. Run both models in the same process (or use MPS/shared CUDA context) to avoid duplicate CUDA context overhead
4. Export TensorRT engine ON the Jetson (not on PC) -- engine is device-specific
5. Set `torch.cuda.set_per_process_memory_fraction(0.4)` to prevent either model from consuming all GPU memory
6. Monitor with `tegrastats` during development to track actual memory usage
7. Consider running Whisper on CPU (tiny model is fast enough) and reserving GPU entirely for YOLO

**Detection:** `tegrastats` shows memory climbing above 80%. CUDA allocation errors in logs. Nodes die without clear error (OOM killer).

**Phase relevance:** Integration phase when YOLO and Whisper first run together. Must be benchmarked before the end-to-end integration milestone. Use `tegrastats` logging during integration tests.

**Sources:**
- [Running two TensorRT models in parallel - NVIDIA Forums](https://forums.developer.nvidia.com/t/jetson-orin-nano-running-two-tensorrt-parallel-models-real-time/351674)
- [Concurrent Vision Inference on Jetson](https://arxiv.org/html/2508.08430v1)
- [Ultralytics Jetson Guide](https://docs.ultralytics.com/guides/nvidia-jetson/)

---

### Pitfall 4: Whisper Blocks Audio Recording -- Missed Wake Phrases

**What goes wrong:** The existing `WhisperPhraseDetector._listen_loop()` records a 2-second chunk, then blocks for 1-5 seconds during Whisper inference. During inference, no audio is being recorded. If someone says "come here" during the inference window, it is completely missed.

**Why it happens:** Single-threaded record-then-transcribe loop. `sounddevice.rec()` is synchronous, and Whisper `transcribe()` is synchronous. The gap between chunks is the inference duration -- potentially 3-5 seconds on Jetson for base model.

**Consequences:** Wake phrase detection is unreliable. Users must repeat themselves multiple times. Effective detection rate drops to ~30-40% because the "listening window" is only 2 out of every 5-7 seconds.

**Prevention:**
1. Split into two threads: recording thread writes continuously to a ring buffer, inference thread reads from the ring buffer
2. Use `sounddevice.InputStream` with a callback instead of blocking `rec()` -- the callback appends to a thread-safe queue
3. Use overlapping windows: feed 2-second chunks that overlap by 1 second so no phrase falls entirely in a gap
4. Use VAD (Voice Activity Detection) from the ReSpeaker's built-in DSP to trigger recording only when speech is detected, reducing unnecessary inference

**Detection:** Test by saying "come here" at random intervals while watching Whisper logs. If detection is inconsistent, this is the cause. Log timestamps of record-start and record-end to measure actual gaps.

**Phase relevance:** Audio integration phase. The existing code (identified in CONCERNS.md) already has this bug. Must be fixed before any end-to-end testing.

**Sources:**
- Identified in project's own CONCERNS.md (Performance Bottlenecks section)

---

## Moderate Pitfalls

### Pitfall 5: pyusb Permission Errors on Jetson for ReSpeaker DOA

**What goes wrong:** Calling `usb.core.find()` for the ReSpeaker (VID 0x2886, PID 0x0018) returns `USBError: [Errno 13] Access denied (insufficient permissions)` even though udev rules are in place. The node must run as root or udev rules must be precisely correct.

**Why it happens:** Udev rules may not cover all USB interfaces on the device. The ReSpeaker exposes multiple USB interfaces (UAC audio, HID for DOA/LED, DFU for firmware). The udev rule must match the top-level device, not a specific interface. Also, udev rules require `udevadm control --reload-rules && udevadm trigger` after changes, and the device must be re-plugged.

**Prevention:**
1. Verify the udev rule matches: `SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0018", MODE="0666"` (not `SUBSYSTEMS` plural, not interface-specific)
2. After adding the rule: `sudo udevadm control --reload-rules && sudo udevadm trigger`
3. Unplug and re-plug the ReSpeaker
4. Test with `python3 -c "import usb.core; d = usb.core.find(idVendor=0x2886, idProduct=0x0018); print(d)"` as the non-root user that ROS nodes run as
5. If all else fails, add the ROS user to the `plugdev` group

**Detection:** Node crashes immediately on startup with permission error. Easy to catch but wastes time if not anticipated.

**Phase relevance:** ReSpeaker DOA integration phase. Validate permissions as the very first step before writing any DOA code.

**Sources:**
- [Seeed Studio Forum - Jetson Mic Array compatibility](https://forum.seeedstudio.com/t/jetson-nano-and-mic-array-v2-0-compatibility/257479)

---

### Pitfall 6: ReSpeaker DOA (pyusb HID) and Audio (ALSA) Interface Confusion

**What goes wrong:** Developers assume pyusb and sounddevice access the same USB interface and worry about contention. In reality, the ReSpeaker exposes separate USB interfaces: UAC 1.0 for audio (accessed by ALSA/sounddevice) and a vendor-specific HID interface for DOA/LED/parameter control (accessed by pyusb). However, the two libraries can still interfere if pyusb claims the wrong interface or if the kernel driver is not properly detached.

**Why it happens:** The ReSpeaker appears as one USB device but exposes multiple interfaces. pyusb's `find()` returns the device-level handle, and calling `set_configuration()` or `claim_interface()` on the wrong interface can steal it from the kernel ALSA driver, killing audio.

**Prevention:**
1. For DOA reads, use only `ctrl_transfer()` on the vendor-specific interface -- do not call `set_configuration()` or `claim_interface()` on the audio interface
2. Access DOA via the HID register read pattern from `tuning.py`: `dev.ctrl_transfer(usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE, 0, id, 0x80, length)`
3. Keep pyusb and sounddevice in the same process to avoid device contention between processes
4. Test concurrent access early: run DOA polling in one thread while recording audio in another

**Detection:** Audio recording suddenly fails with "device busy" or returns silence after DOA code starts. Or DOA reads return stale values while audio is recording.

**Phase relevance:** ReSpeaker integration phase. Both DOA and audio must work simultaneously -- test this specifically before building higher-level logic on top.

**Sources:**
- [respeaker-xmos-hid example](https://github.com/bwhitman/respeaker-xmos-hid/blob/master/listen_and_get_position.py)
- [ReSpeaker USB 4 Mic Array repo](https://github.com/respeaker/usb_4_mic_array)

---

### Pitfall 7: TensorRT Engine Not Portable Between Devices

**What goes wrong:** Developer exports YOLO to TensorRT on their PC (x86 + RTX GPU), then copies the `.engine` file to Jetson Orin NX. The engine fails to load or produces garbage results because TensorRT engines are device-specific (architecture, CUDA version, TensorRT version must all match).

**Why it happens:** TensorRT optimizes for the specific GPU architecture, memory layout, and available tensor cores. An engine built for Ada Lovelace (RTX 4000) is incompatible with Ampere (Orin NX).

**Prevention:**
1. Always export the TensorRT engine ON the Jetson itself: `yolo export model=yolov8n.pt format=engine device=0`
2. Version-lock TensorRT: use the version that ships with JetPack on the Jetson (currently TensorRT 8.6 on JetPack 6.x)
3. Export takes 5-20 minutes on Jetson -- do it once and cache the `.engine` file
4. Store the `.engine` file alongside the `.pt` model with a naming convention that includes the device: `yolov8n_orin_nx_fp16.engine`

**Detection:** Engine load fails with version mismatch error. Or model loads but returns nonsensical detections.

**Phase relevance:** YOLO/perception phase. Export must happen on-device before any detection testing.

**Sources:**
- [Ultralytics Jetson Guide](https://docs.ultralytics.com/guides/nvidia-jetson/)

---

### Pitfall 8: Whisper Transcribes Noise as "Come Here" (False Positives)

**What goes wrong:** Whisper base model, when given ambient robot noise (motor whine, fan noise, footstep vibrations), occasionally hallucinates transcriptions including common phrases. The robot triggers spontaneously without anyone speaking.

**Why it happens:** Whisper is trained on human speech and tends to "find" speech patterns in noise. The ReSpeaker's built-in noise suppression helps but motor vibration conducted through the chassis creates low-frequency noise the DSP does not fully remove. Whisper's confidence scores for hallucinated text can be surprisingly high (0.5-0.7).

**Prevention:**
1. Use the ReSpeaker's built-in VAD (Voice Activity Detection) as a gate -- only send audio to Whisper when VAD reports speech activity
2. Require confidence > 0.8 for wake phrase detection (not the default 0.5)
3. Add a secondary check: require the phrase to appear in 2 consecutive chunks before triggering
4. Use Whisper's `no_speech_probability` output -- if > 0.5, discard the transcription regardless of text match
5. Run Whisper in English-only mode (`model.en`) to reduce hallucination in other languages
6. The ReSpeaker's beamformed channel (channel 0) has much better SNR than raw channels -- ensure sounddevice records from channel 0

**Detection:** Robot triggers approach behavior when no one is speaking. Log all transcriptions with confidence scores during idle periods to measure false positive rate.

**Phase relevance:** Audio integration phase and end-to-end testing phase. Must be tuned during real hardware testing with the robot's motors running.

---

### Pitfall 9: ROS 2 CycloneDDS QoS Mismatch Drops Camera Frames

**What goes wrong:** The GO2's camera publishes `/camera/image_raw` with BEST_EFFORT reliability and KEEP_LAST depth 1. If the YOLO subscriber uses the default QoS (RELIABLE), no messages are received. No error is logged -- the topic simply appears empty.

**Why it happens:** ROS 2 DDS QoS compatibility rules silently drop connections when publisher and subscriber QoS are incompatible. RELIABLE subscriber cannot connect to BEST_EFFORT publisher. This is by design but catches developers off guard because `ros2 topic list` shows the topic exists.

**Prevention:**
1. Always match the publisher's QoS profile: `QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)`
2. Use `ros2 topic info /camera/image_raw -v` to inspect the publisher's actual QoS before writing subscriber code
3. Add a watchdog that logs a warning if no frames received within 5 seconds of subscription

**Detection:** YOLO node runs but never detects anything. `ros2 topic hz /camera/image_raw` shows data flowing, but your node's callback never fires.

**Phase relevance:** YOLO/perception phase. Known from existing project context (PROJECT.md already documents the QoS settings). Verify early.

---

## Minor Pitfalls

### Pitfall 10: sounddevice Selects Wrong ALSA Device on Jetson

**What goes wrong:** Jetson Orin NX may have multiple ALSA devices (HDMI audio, onboard codec, ReSpeaker). `sounddevice.default.device` may default to HDMI output, not the ReSpeaker input. Recording returns silence or errors.

**Prevention:**
1. Enumerate devices: `python3 -c "import sounddevice; print(sounddevice.query_devices())"`
2. Find the ReSpeaker by name (contains "ReSpeaker" or "seeed") and set explicitly: `sd.default.device = (respeaker_index, None)`
3. Alternatively, use the ALSA device name directly: `sd.default.device = 'hw:2,0'` (verify with `arecord -l`)
4. Set device selection in the ROS launch file as a parameter, not hardcoded

**Phase relevance:** Audio integration phase. First thing to validate after sounddevice is installed.

---

### Pitfall 11: Behavior State Machine Floods GO2 with Redundant Commands

**What goes wrong:** The behavior node ticks at 10 Hz and publishes velocity commands every tick even when values have not changed. The GO2's motion controller receives 10 identical commands per second, wasting DDS bandwidth and potentially causing jitter if commands arrive faster than the control loop processes them.

**Prevention:**
1. Only publish when command values change (velocity, direction)
2. Add a minimum time between publishes (rate-limit to 5 Hz for locomotion)
3. Use a dead-man's switch pattern: publish at a steady rate but stop publishing when the robot should stop (the GO2 sport mode halts if no commands received for a timeout period)

**Phase relevance:** Locomotion bridge phase. Already identified in CONCERNS.md.

---

### Pitfall 12: Jetson Has No Internet -- pip Install Fails Silently

**What goes wrong:** The Jetson's network goes through the PC's NAT or a hotspot that blocks DNS. `pip install` hangs or fails. Developer wastes hours debugging when the real issue is network access.

**Prevention:**
1. Pre-download all wheels on the PC: `pip download -d ./wheels <package>`
2. SCP wheels to Jetson: `scp -r ./wheels unitree@192.168.123.18:~/`
3. Install offline: `pip install --no-index --find-links ~/wheels <package>`
4. Maintain a manifest of required packages and their versions for reproducible offline installs
5. For packages requiring compilation (CTranslate2), build on Jetson with source code SCP'd over

**Phase relevance:** Every phase. This is a persistent constraint. Document the offline install procedure in the first phase and reuse throughout.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| ReSpeaker DOA | Pitfall 5 (permissions), Pitfall 6 (interface confusion) | Validate pyusb + sounddevice concurrent access first |
| Whisper on Jetson | Pitfall 2 (CTranslate2 ARM), Pitfall 4 (blocking loop) | Build CTranslate2 from source or use whisper_trt; refactor to ring buffer |
| YOLO on Jetson | Pitfall 7 (engine portability), Pitfall 3 (GPU memory) | Export TensorRT on-device; use nano model with FP16 |
| GO2 Locomotion | Pitfall 1 (sport mode collision) | Use sport service API or properly disable sport mode first |
| End-to-end Integration | Pitfall 3 (GPU OOM), Pitfall 8 (false positives) | Monitor tegrastats; tune confidence thresholds with motors running |
| All Phases | Pitfall 12 (no internet) | Maintain offline wheel cache on PC |

---

## Sources

- [How to disable sport mode on the Go2 - Eric Plass](https://ric.engineering/posts/Unitree-Sportmode/)
- [Sport Mode Control - unitree_ros2 DeepWiki](https://deepwiki.com/unitreerobotics/unitree_ros2/4.2-sport-mode-control)
- [Unitree High Level Sports Service Interface](https://support.unitree.com/home/en/developer/sports_services)
- [CTranslate2 issue on Jetson - NVIDIA Forums](https://forums.developer.nvidia.com/t/ctranslate-2-issue/335263)
- [whisper_trt - NVIDIA-AI-IOT](https://github.com/NVIDIA-AI-IOT/whisper_trt)
- [Running two TensorRT models in parallel - NVIDIA Forums](https://forums.developer.nvidia.com/t/jetson-orin-nano-running-two-tensorrt-parallel-models-real-time/351674)
- [Concurrent Vision Inference on Jetson - arXiv](https://arxiv.org/html/2508.08430v1)
- [Ultralytics YOLO on Jetson Guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- [Seeed Forum - Jetson Mic Array compatibility](https://forum.seeedstudio.com/t/jetson-nano-and-mic-array-v2-0-compatibility/257479)
- [ReSpeaker USB 4 Mic Array repo](https://github.com/respeaker/usb_4_mic_array)
- [respeaker-xmos-hid examples](https://github.com/bwhitman/respeaker-xmos-hid)
- [ReSpeaker Mic Array v2.0 Wiki](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/)

---

*Concerns audit: 2026-04-06*
