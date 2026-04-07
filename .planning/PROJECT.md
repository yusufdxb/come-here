# come-here

## What This Is

A ROS 2 system for the Unitree GO2 robot that responds to the voice command "come here." The robot hears the phrase via a ReSpeaker Mic Array v2.0, estimates the speaker's direction, rotates toward them, visually detects and locks on using YOLO on the GO2's camera, and walks to them. Runs entirely on a Jetson Orin NX mounted on the robot.

## Core Value

The robot hears "come here" and physically comes to the speaker. Sound gets it facing the right direction, vision gets it to the right person.

## Requirements

### Validated

- ✓ ROS 2 workspace with 5 packages (msgs, audio, perception, behavior, bringup) — existing
- ✓ ABC interfaces for AudioDirectionProvider, WakePhraseDetector, PersonDetector — existing
- ✓ Mock implementations for all providers — existing
- ✓ WhisperPhraseDetector (faster-whisper + HF/LoRA backends) — existing
- ✓ Behavior state machine (IDLE → LISTENING → TURN_TO_SOUND → SEARCH_FOR_PERSON → APPROACH_PERSON → STOP) — existing
- ✓ ReSpeaker Mic Array v2.0 recognized on Jetson (DOA confirmed via tuning.py) — verified 2026-04-06

### Active

- [ ] Real AudioDirectionProvider using ReSpeaker DOA via pyusb
- [ ] Live mic audio input to Whisper wake phrase detector (sounddevice + ReSpeaker)
- [ ] Real PersonDetector using YOLO on GO2 camera feed (/camera/image_raw)
- [ ] GO2 locomotion bridge (cmd_rotate/cmd_move → Unitree SDK or cmd_vel)
- [ ] End-to-end integration: hear → rotate → detect → approach → stop
- [ ] Migration from std_msgs placeholders to come_here_msgs custom messages

### Out of Scope

- Whisper fine-tuning — get base model working first, fine-tune in a future milestone
- Multi-person selection — approach the nearest/loudest person, no choosing
- Obstacle avoidance — assume clear path for v1
- Remote monitoring/telemetry — this is standalone on-robot behavior
- PC-side processing — everything runs on the Jetson

## Context

**Hardware:**
- Unitree GO2 robot (192.168.123.161)
- Jetson Orin NX payload (192.168.123.18, WiFi at 172.20.10.6 via hotspot)
- Seeed ReSpeaker Mic Array v2.0 (XMOS XVF3000, USB, VID 0x2886 PID 0x0018)
- GO2 camera accessible as ROS 2 topic /camera/image_raw via CycloneDDS

**Software environment:**
- Ubuntu 22.04 / L4T r36.4 on Jetson
- ROS 2 Humble
- Python 3.10
- CycloneDDS as RMW implementation
- pyusb 1.3.1 installed, udev rules in place for ReSpeaker

**ReSpeaker SDK:**
- Cloned to ~/usb_4_mic_array on Jetson
- tuning.py confirmed working (DOAANGLE returns degrees 0-359)
- Fixed tostring→tobytes for Python 3.10 compatibility
- Built-in DSP: AEC, VAD, DOA, beamforming, noise suppression at 16kHz

**Camera access:**
- /camera/image_raw (sensor_msgs/Image) via CycloneDDS
- Existing viewer script at ~/ros2_ws/view_camera.sh (PC side)
- QoS: BEST_EFFORT reliability, KEEP_LAST history, depth 1

**Existing code state:**
- All nodes work in mock mode
- Real provider slots raise NotImplementedError (designed for plug-in)
- Whisper detector implemented but untested on hardware
- No locomotion bridge exists yet

## Constraints

- **Compute**: Jetson Orin NX — must run YOLO + Whisper concurrently within GPU/CPU budget
- **Network**: Jetson has no reliable DNS (hotspot blocks port 53) — pip installs must be scp'd from PC
- **Audio**: ReSpeaker is UAC1.0 at 16kHz — sounddevice must use the correct ALSA device
- **Latency**: DOA → rotation should feel responsive (<1s from voice to movement start)
- **Deployment**: All code must be deployable to Jetson via scp from T7 SSD

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| ReSpeaker Mic Array v2.0 for audio | Built-in DOA, beamforming, 16kHz, USB plug-and-play on Jetson | — Pending |
| Base Whisper first, fine-tune later | Get end-to-end working before optimizing wake phrase accuracy | — Pending |
| Sound + vision (not sound-only) | DOA gives rough direction, YOLO gives precise person lock-on | — Pending |
| Unitree SDK or cmd_vel for locomotion | Check what's available on GO2, use whichever works | — Pending |
| All processing on Jetson | Standalone operation, no PC dependency at runtime | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-06 after initialization*
