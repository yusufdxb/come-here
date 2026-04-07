# Requirements: come-here

**Defined:** 2026-04-06
**Core Value:** The robot hears "come here" and physically comes to the speaker

## v1 Requirements

### Audio Direction

- [ ] **AUD-01**: ReSpeakerDOAProvider reads DOA angle via pyusb DOAANGLE register at 10 Hz
- [ ] **AUD-02**: DOA readings gated by VOICEACTIVITY register (only report when voice detected)
- [ ] **AUD-03**: DOA degrees converted to ROS radians with configurable robot frame offset parameter
- [ ] **AUD-04**: Direction confidence weighted by VAD strength (higher VAD → higher confidence)

### Wake Phrase Detection

- [ ] **WAKE-01**: Live audio captured from ReSpeaker ALSA device at 16kHz mono (beamformed channel 0)
- [ ] **WAKE-02**: Whisper base.en detects "come here" phrase with confidence threshold > 0.8
- [ ] **WAKE-03**: Listen loop refactored to ring buffer + separate inference thread (no dropped audio)
- [ ] **WAKE-04**: Motor noise false positives suppressed via no_speech_probability > 0.5 check

### Person Detection

- [ ] **VIS-01**: YOLO11n model exported to TensorRT FP16 on Jetson (one-time setup)
- [ ] **VIS-02**: YoloPersonDetector subscribes to /camera/image_raw with BEST_EFFORT QoS, depth 1
- [ ] **VIS-03**: Person bearing estimated from bounding box center normalized to camera HFOV
- [ ] **VIS-04**: Distance estimated from bounding box height; robot stops at ~0.8m
- [ ] **VIS-05**: Approach speed proportional to distance (slower when close, faster when far)

### Locomotion

- [ ] **LOC-01**: LocomotionBridge node translates cmd_rotate/cmd_move to sport_client.Move(vx, vy, vyaw)
- [ ] **LOC-02**: 500ms safety command timeout sends zero velocities on stale commands
- [ ] **LOC-03**: Sport mode API used correctly (never bypass, never fight active sport mode)

### Integration

- [ ] **INT-01**: come_here_msgs replaces all std_msgs placeholders (AudioDirection, WakePhrase, PersonDetection)
- [ ] **INT-02**: End-to-end flow works: hear "come here" → rotate toward sound → detect person → approach → stop
- [ ] **INT-03**: GPU memory validated under concurrent YOLO + Whisper via tegrastats

## v2 Requirements

### Audio Enhancements

- **AUD-V2-01**: VAD pre-gate using ReSpeaker SPEECHDETECTED register to skip silent Whisper chunks
- **AUD-V2-02**: Whisper LoRA fine-tuning on recorded "come here" samples from ReSpeaker

### Vision Enhancements

- **VIS-V2-01**: ByteTrack person tracking across frames for smoother following
- **VIS-V2-02**: Person re-identification after occlusion

### Locomotion Enhancements

- **LOC-V2-01**: IMU-based closed-loop rotation controller for precise turning
- **LOC-V2-02**: Velocity ramp-up/ramp-down for smooth natural movement

### System

- **SYS-V2-01**: Audio feedback (bark/beep) when wake phrase detected
- **SYS-V2-02**: whisper_trt for 3x faster inference

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multi-person selection | v1 approaches nearest/loudest person, no choosing |
| Obstacle avoidance | Assume clear path for v1; SLAM/Nav2 is v2+ |
| Remote monitoring | Standalone on-robot behavior, no telemetry |
| PC-side processing | Everything runs on Jetson |
| Fine-tuned Whisper | Get base model working first |
| ODAS / respeaker_ros | ODAS is overkill; respeaker_ros is ROS 1 only |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| AUD-01 | Pending | Pending |
| AUD-02 | Pending | Pending |
| AUD-03 | Pending | Pending |
| AUD-04 | Pending | Pending |
| WAKE-01 | Pending | Pending |
| WAKE-02 | Pending | Pending |
| WAKE-03 | Pending | Pending |
| WAKE-04 | Pending | Pending |
| VIS-01 | Pending | Pending |
| VIS-02 | Pending | Pending |
| VIS-03 | Pending | Pending |
| VIS-04 | Pending | Pending |
| VIS-05 | Pending | Pending |
| LOC-01 | Pending | Pending |
| LOC-02 | Pending | Pending |
| LOC-03 | Pending | Pending |
| INT-01 | Pending | Pending |
| INT-02 | Pending | Pending |
| INT-03 | Pending | Pending |

**Coverage:**
- v1 requirements: 19 total
- Mapped to phases: 0
- Unmapped: 19 ⚠️

---
*Requirements defined: 2026-04-06*
*Last updated: 2026-04-06 after initial definition*
