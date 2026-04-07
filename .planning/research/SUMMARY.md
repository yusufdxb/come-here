# Project Research Summary

**Project:** come-here (Voice-commanded robot approach system)
**Domain:** Quadruped robotics — voice-triggered, vision-guided approach behavior
**Researched:** 2026-04-06
**Confidence:** MEDIUM-HIGH

## Executive Summary

The "come here" system is a hardware integration milestone that wires real sensor drivers into a pre-existing ROS 2 Humble codebase on a Unitree GO2 / Jetson Orin NX platform. The codebase already has the correct architecture: ABC provider interfaces, a behavior state machine, and mock implementations for every component. The work is almost entirely driver integration — not architecture design.

The recommended approach builds three independent paths — audio, vision, and locomotion — and converges them in the existing behavior node:
- **Audio:** ReSpeaker XMOS DOA via pyusb (already verified) + sounddevice capture to faster-whisper base.en on CPU
- **Vision:** YOLO11n with TensorRT FP16 export on-device, subscribing to GO2 camera at /camera/image_raw
- **Locomotion:** New LocomotionBridge node using unitree_sdk2_python sport_client for velocity commands

## Recommended Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| DOA | pyusb + vendored tuning.py | Already verified on Jetson, reads DOAANGLE + VAD registers |
| Audio capture | sounddevice 16kHz ch0 | Standard ALSA integration, beamformed channel |
| Wake phrase | faster-whisper base.en CPU int8 | ~200ms/2s chunk, keeps GPU free for YOLO |
| Person detection | ultralytics YOLO11n TensorRT FP16 | ~15-30ms/frame on Orin NX, export once on-device |
| Locomotion | unitree_sdk2_python sport_client | Sport mode API, never bypass it |
| Camera bridge | cv_bridge + OpenCV | Standard ROS 2, already on JetPack |

**Resource budget:** Whisper on CPU (~300MB RAM), YOLO on GPU (~200MB VRAM). Well within Orin NX capacity.

## Table Stakes Features

- ReSpeaker DOA reading with robot frame alignment calibration
- Live mic capture with correct ALSA device index
- Whisper "come here" detection with VAD gate and confidence > 0.8
- GO2 locomotion bridge with sport mode API and 500ms safety timeout
- YOLO11n person detection with BEST_EFFORT QoS
- Person bearing from bbox center normalized to camera HFOV
- Monocular distance estimation from bbox height — stop at ~0.8m
- Custom come_here_msgs replacing std_msgs placeholders

## Critical Pitfalls

1. **Sport mode actuator collision** — Never publish motion commands while GO2 sport mode is active. Use sport_client.Move() through the sport service. Validate first 30 minutes. Hardware damage risk.

2. **CTranslate2 has no ARM64 pip wheel** — pip install appears to succeed but shared library is missing on Jetson. Test `import ctranslate2` immediately. Build from source or fall back to whisper_trt.

3. **Whisper blocking loop drops wake phrases** — Existing listen loop has 2-5s dead window during inference. Refactor to ring buffer + separate inference thread. Current effective detection rate ~30-40%.

4. **TensorRT engine is device-specific** — Export .engine on Jetson itself. Never copy from x86/RTX.

5. **CycloneDDS QoS mismatch silently drops camera frames** — GO2 camera publishes BEST_EFFORT. A RELIABLE subscriber receives nothing with no error.

## Suggested Phase Structure (5 phases)

| # | Phase | Risk | Rationale |
|---|-------|------|-----------|
| 1 | Locomotion Bridge | HIGH | Highest uncertainty + damage risk; validate sport mode first |
| 2 | ReSpeaker Audio Path | MEDIUM | Independent; pyusb verified; main work is ROS wrapper + Whisper refactor |
| 3 | YOLO Vision Path | LOW-MEDIUM | Independent; TensorRT export on-device; QoS known |
| 4 | Custom Messages | LOW | Coordinated update before full integration |
| 5 | End-to-End Integration | MEDIUM | All paths converge; tune thresholds with motors running |

Phases 2+3 are parallelizable. Phase 1 should go first due to hardware damage risk.

## Research Flags

**Needs research during planning:**
- Phase 1: GO2 sport mode API exact behavior on this firmware version
- Phase 2: CTranslate2 ARM64 build on JetPack 6 / L4T r36.4

**Standard patterns (skip deep research):**
- Phase 3: Ultralytics Jetson guide is authoritative
- Phase 4: Standard ROS 2 custom message workflow
- Phase 5: Tuning work, not research

## Open Questions

- GO2 firmware version compatibility with sport_client.Move()
- CTranslate2 ARM64 build success on this JetPack version
- Camera HFOV measurement (estimated ~120 degrees fisheye)
- ReSpeaker mount orientation → DOA frame-to-body offset
- ALSA device index for ReSpeaker on this Jetson

## Confidence Assessment

| Area | Level | Notes |
|------|-------|-------|
| Stack | HIGH | Most choices installed and verified |
| Features | MEDIUM-HIGH | Table stakes clear from codebase structure |
| Architecture | HIGH | ABC provider pattern confirmed correct |
| Pitfalls | HIGH | All critical pitfalls confirmed by primary sources |

---
*Research completed: 2026-04-06*
*Sources: STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md*
