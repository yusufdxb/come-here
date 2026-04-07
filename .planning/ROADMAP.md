# Roadmap: come-here

## Overview

The robot needs to hear "come here" and walk to the speaker. The existing codebase has the right architecture (ABC providers, behavior state machine, mock implementations) -- the work is wiring real hardware drivers into those slots. Locomotion goes first because it carries the highest damage risk and must be validated before anything else moves the robot. Audio and vision are independent vertical slices that each deliver a complete sensory capability. Integration converges all paths and validates the system runs concurrently on the Jetson.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Locomotion Bridge** - Validate sport mode API and build safe velocity command interface
- [ ] **Phase 2: Audio Path** - ReSpeaker DOA + live Whisper wake phrase detection
- [ ] **Phase 3: Vision Path** - YOLO person detection, bearing, and distance estimation
- [ ] **Phase 4: Integration** - Custom messages, end-to-end flow, GPU resource validation

## Phase Details

### Phase 1: Locomotion Bridge
**Goal**: Robot can safely receive and execute rotation and movement commands via sport mode API
**Depends on**: Nothing (first phase)
**Requirements**: LOC-01, LOC-02, LOC-03
**Success Criteria** (what must be TRUE):
  1. Robot rotates in place when a cmd_rotate message is published
  2. Robot walks forward/backward when a cmd_move message is published
  3. Robot stops within 500ms when commands stop arriving (safety timeout)
  4. No actuator conflicts -- sport mode remains stable throughout operation
**Plans**: TBD

Plans:
- [ ] 01-01: TBD

### Phase 2: Audio Path
**Goal**: Robot can hear "come here" and know which direction the speaker is
**Depends on**: Phase 1
**Requirements**: AUD-01, AUD-02, AUD-03, AUD-04, WAKE-01, WAKE-02, WAKE-03, WAKE-04
**Success Criteria** (what must be TRUE):
  1. DOA angle is published as a ROS topic when someone speaks near the robot
  2. DOA is silent when no voice is detected (VAD gating works)
  3. Saying "come here" triggers a wake phrase detection event with confidence above threshold
  4. Wake phrase detection does not miss utterances during Whisper inference (ring buffer works)
  5. Motor noise does not trigger false wake phrase detections
**Plans**: TBD

Plans:
- [ ] 02-01: TBD

### Phase 3: Vision Path
**Goal**: Robot can detect a person on camera, estimate their bearing and distance, and modulate approach speed
**Depends on**: Phase 1
**Requirements**: VIS-01, VIS-02, VIS-03, VIS-04, VIS-05
**Success Criteria** (what must be TRUE):
  1. YOLO11n runs on the Jetson GPU via TensorRT FP16 and detects people in the camera feed
  2. Person bearing is published as a ROS topic derived from bounding box position in frame
  3. Robot stops approaching when the person is approximately 0.8m away
  4. Approach speed decreases as the robot gets closer to the person
**Plans**: TBD

Plans:
- [ ] 03-01: TBD

### Phase 4: Integration
**Goal**: All components work together -- hear "come here", rotate toward sound, detect person, approach, stop
**Depends on**: Phase 2, Phase 3
**Requirements**: INT-01, INT-02, INT-03
**Success Criteria** (what must be TRUE):
  1. All std_msgs placeholders replaced with typed come_here_msgs (AudioDirection, WakePhrase, PersonDetection)
  2. Full behavior sequence completes: IDLE -> LISTENING -> TURN_TO_SOUND -> SEARCH_FOR_PERSON -> APPROACH_PERSON -> STOP
  3. YOLO and Whisper run concurrently without GPU memory exhaustion (tegrastats confirms)
**Plans**: TBD

Plans:
- [ ] 04-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4
Note: Phases 2 and 3 are independent and could overlap, but execute sequentially for simplicity.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Locomotion Bridge | 0/0 | Not started | - |
| 2. Audio Path | 0/0 | Not started | - |
| 3. Vision Path | 0/0 | Not started | - |
| 4. Integration | 0/0 | Not started | - |
