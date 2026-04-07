# Feature Landscape

**Domain:** Voice-commanded robot approach (quadruped, mic array DOA, Whisper wake phrase, YOLO person following)
**Researched:** 2026-04-06
**Confidence:** MEDIUM-HIGH (well-understood component domains; integration pattern less documented)

## Table Stakes

Features the system must have or the "come here" behavior simply does not work.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **ReSpeaker DOA reading via pyusb** | Without direction of arrival, robot cannot know which way to turn. The XMOS XVF3000 provides DOAANGLE (0-359 degrees) via USB HID — this is the only sound localization source. | Low | tuning.py already confirmed working on Jetson. Wrap pyusb `ctrl_transfer` in `AudioDirectionProvider.get_direction()`. Read is non-blocking, ~1ms. |
| **DOA degree-to-radian conversion with robot frame alignment** | ReSpeaker reports 0-359 degrees in its own frame. Robot needs radians in its body frame (0=forward). Misalignment means robot turns the wrong way. | Low | One-time calibration: mount mic, read DOA while standing in front of robot, compute offset. Store as ROS parameter. |
| **Live mic audio capture via sounddevice** | Whisper needs audio input. ReSpeaker exposes a UAC1.0 ALSA device at 16kHz. Without live capture, no wake phrase detection. | Low-Med | ReSpeaker 6-channel firmware: channel 0 = processed audio (with AEC/NS), channels 1-4 = raw mics. Use channel 0 for Whisper input. Must identify correct ALSA device index on Jetson. |
| **Whisper "come here" detection on live audio** | The trigger that starts the entire behavior. Without it, robot never activates. WhisperPhraseDetector already exists but is untested on hardware. | Med | faster-whisper base.en on Jetson Orin NX: expect ~1-2s per 2s chunk on CPU (int8). GPU inference faster but competes with YOLO. Use VAD filter to skip silence. Latency budget: 2s chunk + ~1s inference = ~3s from speech to detection. |
| **Rotation toward sound source** | After wake phrase, robot must physically turn toward speaker. This is the first locomotion action. Without it, vision search is random. | Med | Publish yaw velocity on cmd_vel (geometry_msgs/Twist). Need: (1) locomotion bridge to GO2, (2) proportional controller to rotate target_angle radians, (3) completion detection (IMU or timed rotation). |
| **GO2 locomotion bridge (cmd_vel or Unitree SDK)** | Every movement command flows through this. No bridge = no motion. The GO2 supports cmd_vel via CycloneDDS when using unitree_ros2 or go2_ros2_sdk. | Med-High | Two paths: (a) direct cmd_vel publish if GO2's built-in ROS 2 interface accepts it, (b) wrap unitree_sdk2 sport client. Test which works on this specific GO2 firmware. EDU model with Jetson should support direct DDS. |
| **YOLO person detection on camera feed** | After rotating toward sound, robot needs to visually lock onto a person. Without detection, cannot approach. | Med | YOLOv8n with TensorRT on Orin NX: ~50-65 FPS at INT8. Subscribe to /camera/image_raw, run inference, publish PersonEstimate. Use "person" class (COCO class 0). |
| **Person bearing estimation from bounding box** | Robot must steer toward detected person, not just know they exist. Bearing = how far off-center the person is in the image. | Low | bearing_rad = atan2(bbox_center_x - image_center_x, focal_length). Requires camera intrinsics (or approximate from known FOV). |
| **Forward approach with steering correction** | Robot must walk toward person while correcting heading. Pure forward motion without steering loses the target. | Med | Simultaneous linear.x (forward speed) + angular.z (steering correction) on cmd_vel. Proportional control: angular.z proportional to person bearing. |
| **Stop at target distance** | Robot must stop when close enough. Without this, robot walks into the person. | Med | Monocular distance estimation from bounding box height: distance = (known_person_height * focal_length) / bbox_height_pixels. Rough but sufficient for 0.5-1.0m stop distance. Calibrate with one measurement. |
| **Lost-person recovery (back to search)** | If person leaves FOV during approach, robot must not freeze. State machine already handles this (APPROACH -> SEARCH_FOR_PERSON). | Low | Already in behavior_node.py. Just needs real detection data flowing. |
| **Search timeout and return to IDLE** | If robot cannot find person after turning, it should give up gracefully rather than spinning forever. | Low | Already implemented: search_timeout_s parameter (default 10s). |
| **Custom ROS 2 messages (come_here_msgs)** | std_msgs/Float64MultiArray is fragile and undocumented. Real messages enforce structure and enable introspection. | Low | Define AudioDirection.msg (azimuth_rad, confidence, stamp), WakePhrase.msg (phrase, confidence, stamp), PersonDetection.msg (bearing_rad, distance_m, confidence, detected). |

## Differentiators

Features that improve the experience but are not required for basic "come here" functionality.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **VAD-gated Whisper inference** | Running Whisper on every 2s chunk wastes compute. Voice Activity Detection (ReSpeaker has built-in VAD via SPEECHDETECTED register, or use silero-vad) gates Whisper to only run when speech is present. Reduces CPU/GPU load by 80%+ during silence. | Low-Med | ReSpeaker SPEECHDETECTED is a single pyusb read. Check before queuing audio for Whisper. Alternatively, the existing `np.max(np.abs(audio)) < 0.01` silence check is a crude VAD already in place. |
| **TensorRT-optimized YOLO export** | YOLOv8n in PyTorch: ~15-20 FPS on Orin NX. With TensorRT INT8: ~50-65 FPS. 3x speedup frees GPU headroom for Whisper or future tasks. | Med | `yolo export model=yolov8n.pt format=engine device=0 half=True imgsz=640`. One-time export on Jetson. Ultralytics handles TensorRT integration. |
| **Whisper on GPU with TensorRT (whisper_trt)** | NVIDIA's whisper_trt runs ~3x faster than PyTorch on Jetson Orin, using ~60% memory. Makes 2s chunks process in <500ms instead of ~1.5s. | Med-High | Requires building whisper_trt on Jetson. Must time-share GPU with YOLO (they don't run simultaneously: Whisper runs once at wake, YOLO runs during approach). Natural temporal separation. |
| **Proportional yaw controller with IMU feedback** | Instead of open-loop "rotate X radians and hope," use GO2's IMU to close the loop. Results in accurate, smooth turning. | Med | Subscribe to IMU topic from GO2, compute heading error, apply P or PD controller on angular.z. Stop when |error| < threshold. |
| **Multi-phrase detection ("come here", "over here", "come to me")** | More natural interaction. Users don't always say the exact phrase. | Low | Already partially implemented: TRIGGER_PHRASES = {"come here", "come over here"}. Add more phrases to the set. Whisper handles variations naturally since it's full transcription, not keyword-only. |
| **Confidence-weighted DOA smoothing** | Single DOA readings can be noisy. Average over 3-5 readings weighted by confidence for more reliable direction. | Low | Circular mean of last N readings where confidence > threshold. Publish smoothed estimate. Simple ring buffer. |
| **Audio feedback (beep/chirp on wake phrase detection)** | User has no way to know robot heard them. A confirmation sound closes the feedback loop. | Low | Play a short WAV via ALSA on the ReSpeaker's speaker output (it has a built-in speaker driver). Or use GO2's built-in speaker. |
| **Bounding box tracking across frames (SORT/ByteTrack)** | Prevents switching between multiple detected people. Locks onto the first person detected after turning and tracks that specific instance. | Med | ByteTrack is lightweight and works well with YOLO detections. Not needed for single-person scenarios but prevents jitter with multiple people in FOV. |
| **Adaptive approach speed** | Slow down as robot gets closer to avoid overshoot. Walk fast when far, creep when near. | Low | Linear interpolation: speed = approach_speed * clamp(distance / slow_distance, 0.2, 1.0). Already have distance estimate. |
| **Head/camera pan during search** | If GO2 supports head/camera pan, sweep during SEARCH_FOR_PERSON state to expand FOV coverage. | Med | Check if GO2 EDU has a controllable camera gimbal. If not, robot must physically rotate to search (already the fallback). |

## Anti-Features

Features to explicitly NOT build for this milestone.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Multi-person selection / identity tracking** | Massively increases complexity (re-ID, face recognition, gesture selection). Out of scope per PROJECT.md. "Approach the nearest/loudest person, no choosing." | Approach the largest bounding box (closest/most prominent person). If multiple people detected, pick highest confidence + largest bbox. |
| **Obstacle avoidance during approach** | Requires depth sensing or lidar, path planning, costmaps. Entire navigation stack for what should be a clear-path demo. PROJECT.md explicitly excludes this. | Assume clear path for v1. User is responsible for clear line of approach. Add safety: stop if person detection lost (already implemented). |
| **Continuous voice command parsing** | Turning this into a general voice assistant is scope creep. One command: "come here." | Single trigger phrase detection, then behavior executes to completion. No mid-behavior voice commands. |
| **Remote monitoring / telemetry dashboard** | PC-side tooling is out of scope. Everything runs standalone on Jetson. | Use `ros2 topic echo` from PC during development for debugging. No production dashboard. |
| **Whisper fine-tuning in this milestone** | PROJECT.md: "get base model working first, fine-tune in a future milestone." Fine-tuning requires training data collection, GPU training, evaluation pipeline. | Use faster-whisper base.en. If accuracy is poor, adjust confidence_threshold and chunk_duration before considering fine-tuning. |
| **Natural language understanding beyond trigger** | Parsing intent, slot-filling, conversational flow. Not needed for a single fixed command. | String matching: if "come here" in transcription.lower(). Already implemented this way. |
| **Autonomous return to origin** | After reaching the person, going back to where it started. Requires odometry, path memory. | Robot stops at person and returns to IDLE. Operator moves robot back manually or issues another command in the future. |
| **SLAM / mapping** | No need to build a map of the environment for a direct approach behavior. | Dead reckoning during approach is sufficient. Robot walks toward what it sees. |

## Feature Dependencies

```
ReSpeaker DOA (pyusb) ──────────────────┐
                                         ├──> DOA-to-radian conversion ──> Rotation command
Live mic capture (sounddevice) ──────────┤
                                         └──> Whisper detection ──> Wake phrase trigger
                                                                          │
                                                                          v
                                                              Behavior: IDLE -> LISTENING
                                                                          │
                                                              DOA confidence met
                                                                          │
                                                                          v
GO2 locomotion bridge ──────────────────────────────> Rotation execution (TURN_TO_SOUND)
                                                                          │
                                                                          v
YOLO person detection ──> Person bearing ──────────> SEARCH_FOR_PERSON -> APPROACH_PERSON
       │                                                                  │
       └──> Distance estimation ──────────────────────> Stop at distance (APPROACH -> STOP)
                                                                          │
Custom messages (come_here_msgs) ── all nodes ──      Back to IDLE ──────┘
```

**Critical path:** GO2 locomotion bridge is the highest-risk dependency. Without it, nothing moves. Should be validated first.

**Parallel work streams:**
1. Audio path: ReSpeaker DOA + sounddevice capture + Whisper integration (can test without locomotion)
2. Vision path: YOLO on camera feed + bearing/distance estimation (can test without locomotion)
3. Motion path: GO2 locomotion bridge + rotation controller (can test without audio/vision)

All three converge in the behavior node, which already exists as a state machine.

## MVP Recommendation

**Minimum for end-to-end demo:**

1. GO2 locomotion bridge (cmd_vel) -- without this, nothing else matters
2. ReSpeaker DOA provider (pyusb read of DOAANGLE)
3. Live mic audio to Whisper wake phrase detector
4. YOLO person detection on /camera/image_raw
5. Monocular distance estimation (bbox height method)
6. Custom come_here_msgs (clean up the std_msgs debt)

**Defer to post-MVP:**
- TensorRT optimization for YOLO and Whisper (works without it, just slower)
- VAD gating (the silence check already provides crude filtering)
- ByteTrack person tracking (single-person assumption is fine for v1)
- Audio feedback (nice but not functional)
- IMU-based rotation controller (timed open-loop rotation works for demo)

**Implementation order rationale:**
Start with locomotion bridge because it is the only component with real uncertainty (which GO2 interface works?). Audio and vision can be developed and tested independently using `ros2 topic pub` to simulate their outputs to the behavior node. Converge last.

## Sources

- [Seeed ReSpeaker Mic Array v2.0 Wiki](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/) - DOA, VAD, firmware channels
- [respeaker/usb_4_mic_array GitHub](https://github.com/respeaker/usb_4_mic_array) - tuning.py, DOAANGLE register
- [NVIDIA whisper_trt](https://github.com/NVIDIA-AI-IOT/whisper_trt) - 3x speedup on Jetson Orin
- [Seeed Studio: Whisper on Jetson](https://wiki.seeedstudio.com/Whisper_on_Jetson_for_Real_Time_Speech_to_Text/) - deployment guidance
- [Ultralytics YOLO Jetson Guide](https://docs.ultralytics.com/guides/nvidia-jetson/) - TensorRT export, benchmarks
- [YOLOv8 Jetson Orin NX Benchmarks](https://www.seeedstudio.com/blog/2023/03/30/yolov8-performance-benchmarks-on-nvidia-jetson-devices/) - 50-65 FPS at INT8
- [Ultralytics Distance Calculation](https://docs.ultralytics.com/guides/distance-calculation/) - monocular distance estimation
- [Unitree GO2 Developer Docs](https://support.unitree.com/home/en/developer) - SDK and ROS 2 interface
- [go2_ros2_sdk (unofficial)](https://github.com/abizovnuralem/go2_ros2_sdk) - cmd_vel support for GO2
- [unitree_ros2 (official)](https://github.com/unitreerobotics/unitree_ros2) - CycloneDDS-based ROS 2 interface
- [Dist-YOLO: Distance Estimation](https://www.mdpi.com/2076-3417/12/3/1354) - monocular distance via bbox
- [Person-Following Telepresence Robot (YOLO + monocular)](http://arqiipubl.com/ojs/index.php/AMS_Journal/article/view/574) - focal length + person width method
- [Human-Following Strategy for Mobile Robots](https://www.sciencedirect.com/science/article/abs/pii/S0921889022002068) - state machine patterns, PID approach
