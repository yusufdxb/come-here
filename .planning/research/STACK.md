# Technology Stack: Hardware Integration Milestone

**Project:** come-here (Voice-commanded approach system)
**Researched:** 2026-04-06
**Mode:** Ecosystem research for hardware integration layer
**Overall Confidence:** MEDIUM-HIGH

## Context

This stack builds ON TOP of the existing ROS 2 Humble workspace. The codebase already has:
- `rclpy` nodes with ABC interfaces for AudioDirectionProvider, WakePhraseDetector, PersonDetector
- `faster-whisper` / `transformers` + `peft` for Whisper inference
- `sounddevice` for mic input
- Mock implementations for all providers
- CycloneDDS as RMW, Python 3.10, Ubuntu 22.04 / L4T r36.4

This document covers ONLY the new libraries needed for real hardware integration.

---

## Recommended Stack

### 1. ReSpeaker DOA (Direction of Arrival)

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| `pyusb` | 1.3.1 (installed) | HID communication with ReSpeaker XMOS chip | Already verified working on Jetson. tuning.py uses pyusb to read DOA register. No alternative needed. | HIGH |
| `usb_4_mic_array/tuning.py` | vendored (modified) | Read DOAANGLE, VAD status, control DSP params | Already cloned to ~/usb_4_mic_array, tobytes fix applied. This IS the DOA API -- there is no pip package. Vendor it into the ROS package. | HIGH |

**How DOA works:** The ReSpeaker XMOS XVF3000 runs onboard DSP that computes DOA continuously. You read it via USB HID control transfer, not audio processing. The tuning.py `Tuning` class wraps this:

```python
import usb.core
from tuning import Tuning

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
mic = Tuning(dev)
direction = mic.direction  # 0-359 degrees
is_voice = mic.is_voice    # VAD boolean
```

**Do NOT use:** `odas` / `odas_ros` -- overkill for single-source DOA when the hardware already computes it. ODAS is for raw mic arrays without onboard DSP.

**Do NOT use:** `respeaker_ros` (furushchev) -- ROS 1 only, unmaintained since 2019. Write a thin ROS 2 wrapper instead (trivial: poll pyusb, publish to topic).

### 2. Live Microphone Audio Capture

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| `sounddevice` | 0.4.6+ (installed) | Stream audio from ReSpeaker ALSA device | Already in the codebase. Works with ALSA/PortAudio on Jetson. The ReSpeaker appears as a standard USB audio device (UAC 1.0). | HIGH |
| `numpy` | (installed) | Audio buffer manipulation | Already a dependency. | HIGH |

**Critical configuration:**
- ReSpeaker default firmware: 1-channel processed audio at 16kHz (ideal for ASR)
- Alternative firmware: 6-channel (ch0=processed, ch1-4=raw mics, ch5=playback)
- Use 1-channel firmware -- Whisper wants single-channel 16kHz, and the onboard beamforming/NS is better than doing it in software
- Find the device index via `sd.query_devices()` -- look for "ReSpeaker" or "XMOS" in the name
- Set `device=<index>, samplerate=16000, channels=1, dtype='float32'`

**Do NOT use:** `pyaudio` -- sounddevice is the modern replacement, already integrated, and has better error handling. PyAudio's portaudio bindings are fragile on ARM64.

### 3. Speech Recognition (Whisper)

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| `faster-whisper` | 1.1.0+ | CTranslate2-based Whisper inference | Already in codebase. 4x faster than vanilla Whisper, lower memory. CTranslate2 has aarch64 pip wheels. Use `base.en` model for wake phrase -- small enough for Orin NX. | MEDIUM |
| `ctranslate2` | 4.5.0+ | Inference backend for faster-whisper | Must install the aarch64 wheel. `pip install ctranslate2` has ARM64 Linux wheels. If the pip wheel fails on JetPack 6/L4T r36.4, build from source with OpenBLAS backend. | MEDIUM |

**Alternative considered: `whisper_trt` (NVIDIA)**
- 3x faster than PyTorch, ~60% memory vs PyTorch
- base.en on Orin Nano: 0.86s for 20s audio
- BUT: less mature API, harder to integrate with existing faster-whisper code
- VERDICT: Start with faster-whisper (already integrated). Switch to whisper_trt only if inference is too slow for the wake-phrase latency budget (<1s).

**Alternative considered: vanilla `openai-whisper`**
- 4x slower than faster-whisper, higher memory
- No benefit over faster-whisper
- VERDICT: Do not use.

**GPU/CPU split strategy:**
- Run Whisper on CPU (faster-whisper with int8 quantization on CPU is fast enough for base.en on short utterances)
- Reserve GPU entirely for YOLO
- This avoids GPU memory contention and CUDA context switching overhead

### 4. Person Detection (YOLO)

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| `ultralytics` | 8.3.x | YOLO model loading, export, inference API | Single pip install gives you model zoo + TensorRT export + inference. YOLO11n is the current sweet spot for edge: fast, accurate enough for person detection. | HIGH |
| YOLO11n | yolo11n.pt -> .engine | Nano model, person detection | 2.6M params. On Orin NX with TensorRT FP16: ~2.5ms/frame (~400 FPS theoretical, realistically 30+ FPS with pre/post processing). Person is COCO class 0 -- no custom training needed. | HIGH |
| TensorRT | 10.x (from JetPack) | GPU-optimized inference engine | Pre-installed with JetPack 6. Export once on the Jetson, run the .engine file. Do NOT export on PC -- engine files are device-specific. | HIGH |

**Export workflow (run on Jetson):**
```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.export(format="engine", half=True)  # FP16 TensorRT
```

**Inference in ROS 2 node:**
```python
model = YOLO("yolo11n.engine")
results = model.predict(cv_image, classes=[0], conf=0.5)  # class 0 = person
```

**Do NOT use:** YOLOv8 -- YOLO11 supersedes it with better accuracy at same speed, same API.

**Do NOT use:** YOLO26 -- too new (Jan 2026), edge deployment docs are sparse, and YOLO11 is battle-tested on Jetson. Revisit in a future milestone.

**Do NOT use:** DeepStream -- adds massive complexity for marginal gain. The ultralytics Python API with TensorRT export is sufficient for single-camera person detection.

**Do NOT use:** `jetson-inference` (dusty-nv) -- different API, older models, unnecessary when ultralytics handles TensorRT natively.

### 5. GO2 Locomotion Control

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| `geometry_msgs/Twist` on `/cmd_vel` | ROS 2 standard | Send velocity commands to GO2 | The GO2 with CycloneDDS natively subscribes to `/cmd_vel` Twist messages for sport mode locomotion. No SDK wrapper needed -- just publish Twist. This is the simplest, most standard approach. | MEDIUM-HIGH |
| `unitree_ros2` (official) | master | Message definitions, sport mode API | Provides `unitree_go::msg` types for low-level control and `/api/sport/request` for sport mode commands. Use for advanced moves (stand, sit, dance) but `/cmd_vel` suffices for walk/rotate. | MEDIUM |

**Locomotion approach:**
1. **Rotation (from DOA):** Publish `Twist(angular.z=speed)` to `/cmd_vel` to rotate toward sound source
2. **Forward approach (from YOLO):** Publish `Twist(linear.x=speed)` to `/cmd_vel` to walk toward detected person
3. **Stop:** Publish zero Twist

**Critical configuration:**
- `ROS_DOMAIN_ID=0` (GO2 default)
- `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` (already set)
- CycloneDDS 0.10.2 (must match GO2's internal version)
- Ethernet connection to GO2 at 192.168.123.161

**Do NOT use:** `go2_ros2_sdk` (abizovnuralem) for locomotion -- it's a full navigation stack (SLAM, Nav2, joystick). Massive dependency overhead for simple velocity commands. The GO2 already speaks `/cmd_vel` natively over CycloneDDS.

**Do NOT use:** WebRTC protocol -- the Jetson is ethernet-connected to the GO2. CycloneDDS over ethernet is lower latency and more reliable.

### 6. Image Transport (Camera to YOLO)

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| `cv_bridge` | ROS 2 Humble | Convert sensor_msgs/Image to OpenCV/numpy | Standard ROS 2 package for image conversion. The camera publishes `/camera/image_raw` as `sensor_msgs/Image`. cv_bridge converts to numpy for YOLO input. | HIGH |
| `opencv-python` | 4.x (from JetPack) | Image manipulation | Pre-installed on JetPack. Used by cv_bridge and ultralytics internally. | HIGH |

**QoS settings (must match camera publisher):**
```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)
```

---

## Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pyusb` | 1.3.1 | ReSpeaker HID control | DOA reading, DSP parameter tuning |
| `sounddevice` | 0.4.6+ | Audio streaming | Mic capture for Whisper |
| `numpy` | 1.24+ | Array ops | Audio buffers, image arrays |
| `ultralytics` | 8.3.x | YOLO inference | Person detection |
| `cv_bridge` | Humble | ROS Image conversion | Camera subscriber |
| `opencv-python` | 4.x | Image processing | Preprocessing, visualization |
| `faster-whisper` | 1.1.0+ | Speech recognition | Wake phrase detection |
| `ctranslate2` | 4.5.0+ | Inference engine | Backend for faster-whisper |

---

## Alternatives Considered (Summary)

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| DOA | pyusb + tuning.py | ODAS | ODAS is for raw mic arrays; ReSpeaker has onboard DOA |
| DOA ROS wrapper | Write custom (thin) | respeaker_ros | ROS 1 only, unmaintained |
| Speech | faster-whisper (CPU) | whisper_trt | whisper_trt is faster but harder to integrate; try faster-whisper first |
| Speech | faster-whisper | openai-whisper | 4x slower, more memory |
| Person detection | YOLO11n + TensorRT | YOLOv8n | YOLO11 is strictly better, same API |
| Person detection | YOLO11n + TensorRT | YOLO26 | Too new, sparse Jetson docs |
| Person detection | ultralytics Python | DeepStream | Unnecessary complexity for single camera |
| Locomotion | /cmd_vel Twist | go2_ros2_sdk | Overkill; GO2 natively accepts Twist |
| Locomotion | CycloneDDS ethernet | WebRTC | Higher latency, less reliable |
| Audio capture | sounddevice | pyaudio | sounddevice is modern, already integrated |

---

## Installation

**On Jetson (offline-capable via scp from PC):**

```bash
# Already installed (from existing workspace):
# pyusb, sounddevice, numpy, faster-whisper, ctranslate2

# New dependencies:
pip install ultralytics==8.3.40  # YOLO11 support, TensorRT export

# cv_bridge comes from ROS 2 Humble apt packages:
sudo apt install ros-humble-cv-bridge

# Vendor tuning.py into come_here_audio package:
cp ~/usb_4_mic_array/tuning.py ~/ros2_ws/src/come_here_audio/come_here_audio/tuning.py

# Export YOLO model to TensorRT (run once on Jetson):
python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='engine', half=True)"
```

**Offline installation (Jetson has no DNS):**
```bash
# On PC with internet:
pip download ultralytics==8.3.40 --platform manylinux2014_aarch64 --python-version 3.10 --only-binary=:all: -d ./wheels/
# scp wheels to Jetson, then:
pip install --no-index --find-links=./wheels/ ultralytics==8.3.40
```

Note: The `yolo11n.pt` weights file (5.4 MB) must also be scp'd to the Jetson before TensorRT export.

---

## GPU/CPU Resource Budget (Jetson Orin NX)

| Component | Runs On | Estimated Resource | Notes |
|-----------|---------|-------------------|-------|
| YOLO11n TensorRT FP16 | GPU | ~200MB VRAM, ~3ms/frame | Primary GPU consumer |
| faster-whisper base.en int8 | CPU | ~300MB RAM, ~0.5s per 2s chunk | Intermittent (only during listening) |
| ReSpeaker DOA polling | CPU | Negligible (USB HID read) | ~10ms per poll |
| sounddevice capture | CPU | ~5MB (audio buffer) | Continuous 16kHz stream |
| ROS 2 nodes + CycloneDDS | CPU | ~200MB RAM | Baseline overhead |
| **Total estimate** | | ~700MB RAM + ~200MB VRAM | Orin NX has 8-16GB unified memory |

The Orin NX has ample headroom. YOLO and Whisper do NOT need to compete for GPU.

---

## Sources

- [ReSpeaker USB 4 Mic Array - GitHub](https://github.com/respeaker/usb_4_mic_array) -- DOA API, tuning.py reference
- [ReSpeaker Mic Array v2.0 - Seeed Wiki](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/) -- Hardware specs, firmware options
- [Ultralytics YOLO Jetson Guide](https://docs.ultralytics.com/guides/nvidia-jetson/) -- YOLO11 TensorRT benchmarks on Jetson
- [Ultralytics YOLO11 Docs](https://docs.ultralytics.com/models/yolo11/) -- Model architecture, export API
- [NVIDIA whisper_trt](https://github.com/NVIDIA-AI-IOT/whisper_trt) -- TensorRT Whisper benchmarks on Jetson
- [faster-whisper - GitHub](https://github.com/SYSTRAN/faster-whisper) -- CTranslate2 Whisper, ARM64 support
- [CTranslate2 Installation](https://opennmt.net/CTranslate2/installation.html) -- ARM64/aarch64 wheel availability
- [Unitree ROS2 - GitHub](https://github.com/unitreerobotics/unitree_ros2) -- Official GO2 ROS 2 interface
- [go2_ros2_sdk - GitHub](https://github.com/abizovnuralem/go2_ros2_sdk) -- Unofficial SDK, /cmd_vel reference
- [GO2 ROS Control Guide](https://sooratilab.github.io/RobotGuides/go2/ros_control/) -- /cmd_vel Twist, sport mode, CycloneDDS config
- [CTranslate2 ARM64 issue](https://forums.developer.nvidia.com/t/ctranslate-2-issue/335263) -- JetPack 6 compatibility notes

---

*Stack research: 2026-04-06*
