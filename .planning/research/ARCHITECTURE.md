# Architecture Patterns

**Domain:** Voice-triggered robot approach system (ROS 2 + Unitree GO2)
**Researched:** 2026-04-06

## Recommended Architecture

The existing pub-sub architecture with ABC provider interfaces is sound. Each real provider plugs into its ABC slot, the nodes publish to the same topics, and the behavior state machine is unchanged. The key architectural question is how each hardware driver feeds data into its provider.

```
 ReSpeaker USB            System ALSA Mic          GO2 Camera (DDS)       GO2 Sport Mode (DDS)
      |                        |                        |                        ^
  [pyusb ctrl_transfer]   [sounddevice stream]   [ROS 2 subscriber]         [DDS publish]
      |                        |                        |                        |
      v                        v                        v                        |
 ReSpeakerDOAProvider    WhisperPhraseDetector    YoloPersonDetector      LocomotionBridge
 (AudioDirectionProvider) (WakePhraseDetector)    (PersonDetector)        (new node)
      |                        |                        |                        ^
      v                        v                        v                        |
   AudioNode               AudioNode              PerceptionNode          BehaviorNode
      |                        |                        |                        |
      v                        v                        v                        |
 /come_here/               /come_here/             /come_here/            /come_here/
  audio_direction           wake_phrase             person_detection       cmd_rotate, cmd_move
```

### Component Boundaries

| Component | Responsibility | Input | Output | Communicates With |
|-----------|---------------|-------|--------|-------------------|
| **ReSpeakerDOAProvider** | Read DOA angle from ReSpeaker XMOS via pyusb, convert to radians, gate on VAD | USB HID control transfer (pyusb) | DirectionEstimate (azimuth_rad, confidence) | AudioNode (called by timer) |
| **WhisperPhraseDetector** | Record audio from ReSpeaker ALSA device, run faster-whisper, detect "come here" | sounddevice InputStream (ALSA hw device) | PhraseDetection (phrase, confidence) via queue | AudioNode (polled via check()) |
| **YoloPersonDetector** | Subscribe to camera, run YOLO inference, compute bearing and distance | /camera/image_raw (sensor_msgs/Image) | PersonEstimate (bearing_rad, distance_m, confidence) | PerceptionNode (called by timer) |
| **LocomotionBridge** | Translate cmd_rotate/cmd_move into GO2 sport mode velocity commands | /come_here/cmd_rotate, /come_here/cmd_move | DDS publish to rt/api/sport/request (vx, vy, vyaw) | BehaviorNode (via topics), GO2 MCU (via DDS) |
| **AudioNode** | Orchestrate DOA provider and wake detector, publish to topics | Provider outputs | /come_here/audio_direction, /come_here/wake_phrase | BehaviorNode (via topics) |
| **PerceptionNode** | Orchestrate person detector, publish to topics | Detector output | /come_here/person_detection | BehaviorNode (via topics) |
| **BehaviorNode** | State machine: IDLE through STOP | All /come_here/ sensor topics | /come_here/cmd_rotate, /come_here/cmd_move, /come_here/state | LocomotionBridge (via topics) |

## Data Flow: Detailed Per-Component

### 1. ReSpeaker DOA -> AudioNode

**How pyusb DOA readings feed into a ROS 2 node:**

The ReSpeaker XMOS firmware exposes DSP parameters via USB HID control transfers. The existing `tuning.py` Tuning class reads DOAANGLE (0-359 degrees, integer) and VOICEACTIVITY (0 or 1) using `usb.core.Device.ctrl_transfer()`. This is a synchronous USB read that returns immediately (sub-millisecond).

The ReSpeakerDOAProvider should:

```python
class ReSpeakerDOAProvider(AudioDirectionProvider):
    """Read DOA from ReSpeaker Mic Array v2.0 via pyusb HID interface."""

    def setup(self):
        import usb.core
        self._dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if self._dev is None:
            raise RuntimeError("ReSpeaker Mic Array v2.0 not found")
        # Import the Tuning class from respeaker SDK
        from tuning import Tuning
        self._tuning = Tuning(self._dev)

    def get_direction(self) -> DirectionEstimate | None:
        vad = self._tuning.is_voice()
        if not vad:
            return None  # No speech detected, skip
        angle_deg = self._tuning.direction  # 0-359, 0=front
        # Convert: ReSpeaker 0=front, clockwise. ROS convention: 0=forward, positive=left (CCW)
        angle_rad = -math.radians(angle_deg)  # negate for CCW convention
        # Normalize to [-pi, pi]
        angle_rad = (angle_rad + math.pi) % (2 * math.pi) - math.pi
        return DirectionEstimate(azimuth_rad=angle_rad, confidence=0.8 if vad else 0.0)

    def teardown(self):
        self._dev = None
```

**Key design decisions:**
- **Poll, do not stream.** The AudioNode timer already ticks at 10 Hz. Each tick calls `get_direction()` which does one USB control transfer (fast). No threading needed for DOA.
- **Gate on VAD.** Return None when VOICEACTIVITY is 0. This prevents the behavior node from acting on stale DOA angles.
- **Confidence is binary-ish.** The XMOS firmware does not expose a DOA confidence value. When VAD is active, use a fixed confidence (0.8). The behavior node's threshold (0.5) will pass this through.
- **Angle convention conversion.** ReSpeaker reports 0-359 clockwise from front. The DirectionEstimate uses radians with positive=left (standard ROS REP-103). Negate and convert.

**Polling rate:** 10 Hz is fine. DOAANGLE updates at the firmware's internal rate (roughly 100ms intervals per XMOS spec). Polling faster than 10 Hz gains nothing.

### 2. Sounddevice Audio Stream -> Whisper Inference

**How sounddevice connects to Whisper:**

The existing WhisperPhraseDetector already implements the correct pattern. The background thread uses `sd.rec()` to capture fixed-length chunks and feeds them to faster-whisper. The changes needed for real hardware:

1. **Identify the correct ALSA device index.** The ReSpeaker appears as a UAC1.0 device. Run `python3 -c "import sounddevice; print(sounddevice.query_devices())"` on the Jetson to find its index. Pass that index as `mic_device` parameter.

2. **Use channel 0 (beamformed output).** The ReSpeaker exposes 6 channels via ALSA (4 raw mics + 1 beamformed + 1 ASR-processed). Channel 0 of the processed output is the beamformed audio with noise suppression already applied by the XMOS DSP. Configure sounddevice for `channels=1` reading from the correct device.

3. **Sample rate must be 16000.** The ReSpeaker is UAC1.0 at 16kHz. faster-whisper also expects 16kHz float32. This already matches the existing code.

**Architecture pattern -- sd.rec() chunked recording (keep as-is):**

```
Background thread loop:
  audio = sd.rec(chunk_samples, samplerate=16000, channels=1, device=RESPEAKER_IDX)
  sd.wait()
  -> skip if silent (max < 0.01)
  -> transcribe with faster-whisper
  -> if "come here" found, enqueue PhraseDetection
  
Main thread (AudioNode._tick):
  detection = self._wake_detector.check()  # non-blocking queue.get_nowait()
  -> if detection, publish to /come_here/wake_phrase
```

**Why sd.rec() over InputStream callback:** The existing approach records a complete chunk, then transcribes. This is simpler and correct for Whisper, which needs complete utterances (not streaming). A callback-based InputStream would add complexity (ring buffer management) with no accuracy benefit. Whisper is not a streaming model -- it processes complete audio segments.

**Critical detail -- device selection at launch:**

```python
# In AudioNode.__init__, add parameter:
self.declare_parameter('mic_device_index', -1)  # -1 = system default

# Pass to WhisperPhraseDetector:
mic_idx = self.get_parameter('mic_device_index').value
mic_device = mic_idx if mic_idx >= 0 else None
self._wake_detector = WhisperPhraseDetector(..., mic_device=mic_device)
```

**Concurrency note:** DOA (pyusb) and audio recording (sounddevice/ALSA) use the ReSpeaker simultaneously. This works because DOA reads from the XMOS control endpoint (USB HID) while audio reads from the audio endpoint (USB isochronous). They are independent USB pipes.

### 3. YOLO Detections from /camera/image_raw

**How YOLO detections drive approach behavior:**

The YoloPersonDetector subscribes to `/camera/image_raw`, runs YOLO inference, computes bearing and distance, and returns a PersonEstimate. The PerceptionNode calls `detect()` on its timer.

**Architecture change: PerceptionNode needs camera data routed to detector.**

Currently, PerceptionNode just calls `self._detector.detect()` with no arguments. The real detector needs the camera image. Recommended approach:

**Detector owns the subscription internally:**

```python
class YoloPersonDetector(PersonDetector):
    def __init__(self, node: Node):
        self._node = node  # Need the node for creating subscriptions
        self._latest_image = None
        self._model = None

    def setup(self):
        from ultralytics import YOLO
        self._model = YOLO("yolo11n.engine")  # TensorRT-exported nano model
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        from sensor_msgs.msg import Image
        self._sub = self._node.create_subscription(
            Image, '/camera/image_raw', self._image_cb, qos
        )

    def _image_cb(self, msg):
        # Convert ROS Image to numpy (manual conversion, no cv_bridge dependency)
        self._latest_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1
        )

    def detect(self) -> PersonEstimate:
        if self._latest_image is None:
            return PersonEstimate(0.0, 0.0, 0.0, detected=False)
        results = self._model(self._latest_image, classes=[0], conf=0.5, verbose=False)
        # Pick highest-confidence person detection
        # Compute bearing and distance from bounding box
        ...
```

**Why the detector owns its subscription:** The subscription QoS (BEST_EFFORT, depth 1) is hardware-specific knowledge that belongs in the detector, not the node. The PerceptionNode stays clean -- it just calls `detect()` on its timer. The image_cb fires asynchronously via ROS spin, updating `_latest_image`. The timer-driven `detect()` reads whatever image is current.

**Bearing computation from bounding box:**

```python
# Camera horizontal FOV for GO2 (approximately 150 degrees fisheye, ~120 effective)
HFOV_RAD = math.radians(120)  # verify with actual camera spec
image_width = frame.shape[1]
box_center_x = (x1 + x2) / 2.0
# Normalize to [-0.5, 0.5] from image center
normalized_x = (box_center_x / image_width) - 0.5
bearing_rad = -normalized_x * HFOV_RAD  # image-left = robot-left = positive bearing
```

**Distance estimation from bounding box height (monocular):**

Without depth sensing, estimate distance using known approximate height of a standing person and the bounding box height in pixels:

```python
PERSON_HEIGHT_M = 1.7  # average standing person
FOCAL_LENGTH_PX = image_height / (2 * math.tan(VFOV_RAD / 2))
box_height_px = y2 - y1
distance_m = (PERSON_HEIGHT_M * FOCAL_LENGTH_PX) / box_height_px
```

This is a rough estimate (accuracy decreases with distance and partial visibility) but sufficient for approach behavior where the robot refines continuously. The stop condition (distance <= 0.8m) will be reliable because at close range the bounding box is large and the estimate is accurate.

**YOLO model selection for Jetson Orin NX:**
- Use YOLO11n (nano) exported to TensorRT for best inference speed
- Expected ~15-30ms per frame on Orin NX with TensorRT, easily supporting 10 Hz detection
- Export once on the Jetson: `yolo export model=yolo11n.pt format=engine device=0`
- Store the .engine file in the models/ directory alongside Whisper models

### 4. Locomotion Bridge to GO2

**The right way to bridge cmd_rotate/cmd_move to Unitree GO2 locomotion:**

The GO2 sport mode accepts velocity commands via DDS topic `rt/api/sport/request` using `sport_client.Move(vx, vy, vyaw)`. Since the Jetson communicates with the GO2 over CycloneDDS (same RMW as the come-here system), the bridge can publish directly to the GO2's DDS topic.

**Architecture: New LocomotionBridge node.**

```python
class LocomotionBridge(Node):
    """Translates come_here behavior commands to GO2 sport mode velocity."""

    def __init__(self):
        super().__init__('locomotion_bridge')

        # Subscribe to behavior commands
        self.create_subscription(Float64, '/come_here/cmd_rotate', self._rotate_cb, 10)
        self.create_subscription(Float64, '/come_here/cmd_move', self._move_cb, 10)

        # GO2 sport mode interface via unitree_sdk2_python
        self._vx = 0.0
        self._vyaw = 0.0
        self._last_cmd_time = self.get_clock().now()

        # Publish at fixed rate to keep sport mode alive
        self._cmd_timer = self.create_timer(0.1, self._send_cmd)  # 10 Hz

    def _rotate_cb(self, msg):
        # Convert target azimuth to yaw velocity (proportional controller)
        self._vyaw = self._clamp(msg.data * 1.0, -1.0, 1.0)
        self._last_cmd_time = self.get_clock().now()

    def _move_cb(self, msg):
        self._vx = self._clamp(msg.data, 0.0, 0.5)  # forward only, max 0.5 m/s
        self._last_cmd_time = self.get_clock().now()

    def _send_cmd(self):
        # Safety: zero velocity if no command received recently
        now = self.get_clock().now()
        if (now - self._last_cmd_time).nanoseconds / 1e9 > 0.5:
            self._vx = 0.0
            self._vyaw = 0.0
        self._sport_client.Move(self._vx, 0.0, self._vyaw)
```

**Which interface to use -- investigate in order:**

| Interface | Pros | Cons | Priority |
|-----------|------|------|----------|
| **unitree_sdk2_python sport_client** | Official, direct DDS, no extra dependencies | Requires unitree_sdk2_python installed | Try first -- most reliable |
| **go2_ros2_sdk /cmd_vel (Twist)** | Standard ROS 2, easy | Unofficial, extra node running | Fallback |
| **Direct DDS publish to rt/api/sport/request** | No SDK dependency | Must match exact message format, fragile | Last resort |

**Recommendation:** Use `unitree_sdk2_python` with `sport_client.Move(vx, vy, vyaw)`. This is the official Unitree SDK, communicates via CycloneDDS (already the RMW for the whole system), and gives direct sport mode control.

**Why a separate node (not inside BehaviorNode):**
- BehaviorNode publishes abstract commands (rotate by X radians, move at Y speed). The bridge translates to hardware-specific protocol.
- If the GO2 SDK interface changes, only the bridge changes.
- The bridge can enforce safety limits (max velocity, timeout to zero if no command received).
- The bridge can be tested independently (manual cmd_rotate/cmd_move publishes).

**Safety: command timeout.**

The bridge must zero velocities if it has not received a command within a timeout (500ms). This prevents the robot from walking forever if the behavior node crashes.

## Complete Topic Map

```
Topic                           Type                        Publisher         Subscriber         QoS
-----------------------------------------------------------------------------------------------
/camera/image_raw               sensor_msgs/Image           GO2 (external)   YoloPersonDetector BEST_EFFORT, depth 1
/come_here/audio_direction      Float64MultiArray*          AudioNode         BehaviorNode       RELIABLE, depth 10
/come_here/wake_phrase          String*                     AudioNode         BehaviorNode       RELIABLE, depth 10
/come_here/person_detection     Float64MultiArray*          PerceptionNode    BehaviorNode       RELIABLE, depth 10
/come_here/cmd_rotate           Float64                     BehaviorNode      LocomotionBridge   RELIABLE, depth 10
/come_here/cmd_move             Float64                     BehaviorNode      LocomotionBridge   RELIABLE, depth 10
/come_here/state                String                      BehaviorNode      (debug/monitor)    RELIABLE, depth 10
/come_here/mock_trigger         Bool                        (test tool)       AudioNode          RELIABLE, depth 10
/come_here/mock_person          Bool                        (test tool)       PerceptionNode     RELIABLE, depth 10
rt/api/sport/request            unitree sport msg           LocomotionBridge  GO2 MCU            (DDS native)

* Will migrate to come_here_msgs types (AudioDirection, WakePhrase, PersonDetection)
```

## Patterns to Follow

### Pattern 1: ABC Provider Implementation with Hardware Init in setup()

**What:** All hardware initialization (USB device find, model loading, subscription creation) happens in `setup()`, not `__init__()`. This allows the node to construct the provider, configure parameters, then call setup.

**When:** Every real provider implementation.

**Why:** If setup fails (device not found, model file missing), the error is clear and happens at a known point. The node can catch it and fall back to mock or log a useful error.

### Pattern 2: Timer-Driven Polling (not Event-Driven) for Sensor Fusion

**What:** The AudioNode and PerceptionNode tick at a fixed rate and poll their providers. They do not publish on every sensor event.

**When:** All sensor nodes.

**Why:** The behavior node needs predictable update rates for its state machine timing (search timeout, approach control loop). Event-driven publishing from hardware callbacks would have unpredictable rates and could flood the behavior node. The 10 Hz tick rate is fast enough for human-speed interaction.

### Pattern 3: Separate Locomotion Bridge Node

**What:** A dedicated node subscribes to abstract motion commands and translates to hardware-specific protocol.

**When:** Any robot locomotion interface.

**Why:** Decouples behavior logic from hardware SDK. Enables safety features (command timeout, velocity limits) in one place. Allows testing behavior without a robot.

### Pattern 4: Background Thread for Blocking I/O (Whisper Only)

**What:** WhisperPhraseDetector runs audio recording + inference in a daemon thread, communicating results via thread-safe queue.

**When:** Only for operations that block for significant time (>100ms). In this system, only Whisper inference qualifies (~200-500ms per chunk on Jetson).

**Why:** sd.rec() + sd.wait() blocks for the full chunk duration (2s). Running this in the main ROS spin thread would prevent timer callbacks from firing.

**Do NOT use this pattern for:** DOA reading (sub-ms), YOLO inference (15-30ms on TensorRT -- fast enough in the timer callback).

## Anti-Patterns to Avoid

### Anti-Pattern 1: Running YOLO in a Background Thread

**What:** Putting YOLO inference in a separate thread like WhisperPhraseDetector.

**Why bad:** YOLO on TensorRT runs in 15-30ms -- well within a 100ms timer tick. Threading adds complexity (GIL contention, frame synchronization) with no benefit. The PerceptionNode timer already handles the rate.

**Instead:** Run YOLO synchronously in the timer callback via `detect()`.

### Anti-Pattern 2: Publishing DOA on Every USB Poll Without VAD Gate

**What:** Always publishing an audio_direction message, even when no one is speaking.

**Why bad:** The DOAANGLE register holds the last computed angle, even when no sound is present. Publishing it continuously would cause the behavior node to act on stale/random directions.

**Instead:** Check `is_voice()` (VOICEACTIVITY) first. Return None from `get_direction()` when VAD is inactive.

### Anti-Pattern 3: Using geometry_msgs/Twist for Internal Behavior Commands

**What:** Having BehaviorNode publish Twist messages to /cmd_vel directly.

**Why bad:** Couples the behavior logic to a specific locomotion interface. Makes testing harder (need to interpret Twist). Prevents adding safety layers.

**Instead:** Keep abstract cmd_rotate (radians) and cmd_move (m/s) topics. Let the LocomotionBridge translate.

### Anti-Pattern 4: Loading YOLO Model on Every detect() Call

**What:** Creating the YOLO model instance inside `detect()`.

**Why bad:** Model loading takes seconds. TensorRT engine compilation takes minutes on first run.

**Instead:** Load once in `setup()`. Keep the model in memory. Export to TensorRT engine format ahead of time (during deployment, not at runtime).

### Anti-Pattern 5: GPU Contention Between YOLO and Whisper

**What:** Running both YOLO (TensorRT) and Whisper (CUDA) on GPU simultaneously.

**Why bad:** CUDA context switching on Orin NX adds latency spikes. Shared GPU memory pressure.

**Instead:** Run faster-whisper on CPU with int8 compute type (already the default). Reserve GPU exclusively for YOLO. The base.en model on CPU int8 transcribes a 2s chunk in ~200ms on Orin NX -- fast enough.

### Anti-Pattern 6: Nav2 for Simple Approach

**What:** Integrating the full Nav2 stack for walk-to-person behavior.

**Why bad:** Nav2 requires a map, localization, costmaps, planners -- massive overhead for "walk forward and stop."

**Instead:** Direct proportional control from bounding box bearing and distance. The behavior state machine already handles this.

## Suggested Build Order (Dependencies Between Components)

The following order reflects actual technical dependencies:

```
Phase 1: ReSpeaker DOA Provider
  Dependencies: pyusb (installed), tuning.py (cloned), udev rules (done)
  No other component depends on changes here
  Can test independently: publish to /come_here/audio_direction, verify angles

Phase 2: Whisper on Live Mic
  Dependencies: sounddevice (needs ALSA device index), faster-whisper model files
  Depends on: ReSpeaker being accessible as ALSA device (independent of Phase 1 pyusb)
  Can test independently: speak "come here", verify /come_here/wake_phrase publishes

Phase 3: YOLO Person Detector
  Dependencies: ultralytics, /camera/image_raw topic from GO2
  Independent of Phases 1-2
  Can test independently: point camera at person, verify /come_here/person_detection
  Needs: camera FOV calibration for bearing, distance estimation tuning

Phase 4: Locomotion Bridge
  Dependencies: unitree_sdk2_python, GO2 sport mode access
  Independent of Phases 1-3 for development (test with manual topic publishes)
  Can test independently: publish cmd_rotate/cmd_move, verify robot moves

Phase 5: End-to-End Integration
  Dependencies: ALL of Phases 1-4
  Tuning: confidence thresholds, approach speed, stop distance
  This is where the state machine gets exercised with real sensor data

Phase 6: Message Migration (std_msgs -> come_here_msgs)
  Can happen at any point but most efficient after integration works
  All nodes and behavior need coordinated update
```

**Parallelism:** Phases 1+2 can run in parallel with Phase 3. Phase 4 can start anytime. The critical path is whichever of {1+2, 3, 4} takes longest, then Phase 5.

## Compute Budget

| Resource | YOLO11n (TensorRT) | faster-whisper base.en (CPU int8) | ReSpeaker DOA (pyusb) | Total |
|----------|-------------------|----------------------------------|----------------------|-------|
| GPU | ~200MB VRAM, 15-30ms/frame | None (CPU only) | None | ~200MB |
| CPU | Minimal (pre/post) | ~1 core during 2s inference | Negligible | ~1 core peak |
| USB | None | Isochronous audio endpoint | HID control endpoint | Both from same device |
| DDS network | /camera/image_raw (heaviest) | None | None | One image stream |

The Orin NX (16GB, 6-8 CPU cores) handles this comfortably.

## Sources

- [ReSpeaker USB 4 Mic Array - GitHub (tuning.py, DOA interface)](https://github.com/respeaker/usb_4_mic_array)
- [ReSpeaker Mic Array v2.0 - Seeed Wiki (hardware spec, DSP features)](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/)
- [faster-whisper - PyPI (inference engine, API)](https://pypi.org/project/faster-whisper/)
- [Real-Time Audio Processing - python-sounddevice DeepWiki](https://deepwiki.com/spatialaudio/python-sounddevice/4.3-real-time-audio-processing)
- [Ultralytics YOLO ROS Quickstart (node structure, camera subscription)](https://docs.ultralytics.com/guides/ros-quickstart/)
- [yolo_ros - GitHub (ROS 2 YOLO integration package)](https://github.com/mgonzs13/yolo_ros)
- [Unitree GO2 ROS 2 SDK (unofficial, cmd_vel interface)](https://github.com/abizovnuralem/go2_ros2_sdk)
- [Unitree SDK2 - Official Developer Docs (sport_client.Move)](https://support.unitree.com/home/en/developer)
- [go2_python_sdk - DDS direct control (sport mode topics)](https://github.com/legion1581/go2_python_sdk)
- [Unitree GO2 sport mode control (DDS topic structure)](https://ric.engineering/posts/Unitree-Sportmode/)
- [Ultralytics Distance Calculation (monocular distance estimation)](https://docs.ultralytics.com/guides/distance-calculation/)

---

*Architecture research: 2026-04-06*
