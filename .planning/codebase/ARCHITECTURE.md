# Architecture

**Analysis Date:** 2026-04-06

## Pattern Overview

**Overall:** Publish-subscribe ROS 2 system with abstract provider pattern.

**Key Characteristics:**
- Three independent ROS 2 nodes (audio, perception, behavior) communicate via typed topics
- All sensor inputs (direction, wake phrase, person detection) abstracted behind ABC interfaces
- Behavior driven by a 6-state finite state machine triggered by sensor events
- Default-to-mock approach: all providers have mock implementations, real providers raise `NotImplementedError`
- Non-blocking sensor polling: background threads for audio, synchronous for vision

## Layers

**Message Layer (come_here_msgs):**
- Purpose: Define strongly-typed ROS messages for cross-node communication
- Location: `come_here_msgs/msg/`
- Contains: Three message definitions (AudioDirection, PersonDetection, WakePhrase)
- Depends on: `std_msgs`
- Used by: All three functional nodes

**Audio Perception (come_here_audio):**
- Purpose: Detect wake phrase and estimate sound source direction
- Location: `come_here_audio/come_here_audio/`
- Contains: Two provider abstractions (AudioDirectionProvider, WakePhraseDetector) and a ROS node
- Depends on: Audio providers (mock or real)
- Used by: Behavior node via `/come_here/wake_phrase` and `/come_here/audio_direction` topics

**Visual Perception (come_here_perception):**
- Purpose: Detect and localize a person in the camera frame
- Location: `come_here_perception/come_here_perception/`
- Contains: PersonDetector abstraction and perception ROS node
- Depends on: Person detection provider (mock or real)
- Used by: Behavior node via `/come_here/person_detection` topic

**Behavior Execution (come_here_behavior):**
- Purpose: State machine that orchestrates the "come here" sequence
- Location: `come_here_behavior/come_here_behavior/`
- Contains: State enum and BehaviorNode that transitions between IDLE, LISTENING, TURN_TO_SOUND, SEARCH_FOR_PERSON, APPROACH_PERSON, STOP
- Depends on: Wake phrase, audio direction, person detection subscriptions
- Used by: Robot locomotion layer (cmd_rotate, cmd_move topics)

**System Launch (come_here_bringup):**
- Purpose: Unified launch and configuration entry point
- Location: `come_here_bringup/launch/`
- Contains: come_here.launch.py that starts all three nodes with shared parameters
- Depends on: All functional packages
- Used by: End user to start the system

## Data Flow

**Wake Phrase Detection Path:**

1. Audio thread (WhisperPhraseDetector or mock) checks microphone continuously
2. When "come here" detected, publishes to `/come_here/wake_phrase` (String or WakePhrase msg)
3. BehaviorNode receives wake_phrase event via callback
4. If in IDLE state, transitions to LISTENING

**Audio Direction Path:**

1. Audio thread (AudioDirectionProvider subclass) estimates direction of dominant sound source
2. Publishes direction estimate to `/come_here/audio_direction` (Float64MultiArray: [azimuth_rad, confidence])
3. BehaviorNode receives direction via callback, updates `_last_azimuth`, `_last_dir_confidence`
4. In LISTENING state, when confidence exceeds threshold (default 0.5), transitions to TURN_TO_SOUND

**Person Detection Path:**

1. Perception node (PersonDetector subclass) polls camera at fixed rate (10 Hz default)
2. Publishes detection result to `/come_here/person_detection` (Float64MultiArray: [bearing, distance, confidence, detected])
3. BehaviorNode receives person data via callback, updates bearing/distance/confidence/detected flags
4. In SEARCH_FOR_PERSON state, when detection found with confidence >= threshold, transitions to APPROACH_PERSON

**Behavior Execution:**

1. BehaviorNode ticks at fixed rate (10 Hz default)
2. Each tick evaluates current state and transitions
3. On transitions:
   - LISTENING → TURN_TO_SOUND: Publish cmd_rotate with last_azimuth
   - TURN_TO_SOUND → SEARCH_FOR_PERSON: Immediately transition
   - SEARCH_FOR_PERSON → APPROACH_PERSON: When person found
   - APPROACH_PERSON → continues: Publish cmd_rotate (bearing) and cmd_move (speed)
   - APPROACH_PERSON → STOP: When distance <= 0.8m
   - STOP → IDLE: After confirming motion stopped

**State Management:**

- Audio perception: Thread-safe queue in WhisperPhraseDetector, pull-based via check()
- Visual perception: Mock detector holds state (_detected, _bearing, _distance), modified externally via set_detected()
- Behavior: All state in BehaviorNode (_state, _last_azimuth, _last_dir_confidence, _person_*, _search_start_time)
- No persistent state: System resets to IDLE at completion

## Key Abstractions

**AudioDirectionProvider:**
- Purpose: Decouple ROS node from specific microphone hardware
- Examples: `come_here_audio/audio_direction_provider.py` (ABC), `come_here_audio/mock_audio_provider.py` (mock)
- Pattern: Abstract base class with setup(), get_direction(), teardown(); returns DirectionEstimate dataclass (azimuth_rad, confidence)
- Blocking behavior: None, returns immediately with None if no sound

**WakePhraseDetector:**
- Purpose: Decouple ROS node from specific ASR engine
- Examples: `come_here_audio/wake_phrase_detector.py` (ABC), `come_here_audio/whisper_phrase_detector.py` (real with HF + LoRA support)
- Pattern: Abstract base class with setup(), check(), teardown(); returns PhraseDetection dataclass (phrase, confidence) or None
- Concurrency model: WhisperPhraseDetector runs inference in background thread, check() is non-blocking via queue.Queue

**PersonDetector:**
- Purpose: Decouple ROS node from specific vision model
- Examples: `come_here_perception/person_detector.py` (ABC), stub real implementation pending YOLO/MediaPipe integration
- Pattern: Abstract base class with setup(), detect(), teardown(); returns PersonEstimate dataclass (bearing_rad, distance_m, confidence, detected)
- Blocking behavior: None, returns immediately

## Entry Points

**AudioNode:**
- Location: `come_here_audio/come_here_audio/audio_node.py`
- Triggers: Launched by come_here.launch.py
- Responsibilities: Initialize AudioDirectionProvider and WakePhraseDetector based on parameters; tick at fixed rate to publish audio estimates; handle mock trigger subscriptions

**PerceptionNode:**
- Location: `come_here_perception/come_here_perception/perception_node.py`
- Triggers: Launched by come_here.launch.py
- Responsibilities: Initialize PersonDetector; tick at fixed rate to publish person detection; handle mock person toggle subscriptions

**BehaviorNode:**
- Location: `come_here_behavior/come_here_behavior/behavior_node.py`
- Triggers: Launched by come_here.launch.py
- Responsibilities: Subscribe to wake_phrase, audio_direction, person_detection; tick state machine; publish state, cmd_rotate, cmd_move

**Launch Entry:**
- Location: `come_here_bringup/launch/come_here.launch.py`
- Triggers: `ros2 launch come_here_bringup come_here.launch.py [use_mock:=true|false]`
- Responsibilities: Load config files from all three packages; instantiate and start all three nodes; forward use_mock parameter to each node's config

## Error Handling

**Strategy:** Fail-safe with graceful degradation.

**Patterns:**

- **Missing real providers:** BehaviorNode, AudioNode, PerceptionNode raise NotImplementedError immediately if real provider requested but not implemented. Forces developer to explicitly set use_mock:=true or implement provider.
- **Audio input failures:** WhisperPhraseDetector thread catches audio recording exceptions, logs, sleeps 0.5s, retries. Non-blocking: doesn't crash main thread.
- **Silent audio chunks:** WhisperPhraseDetector skips chunks with max amplitude < 0.01 to avoid processing pure silence.
- **Low-confidence estimates:** Behavior node respects confidence thresholds (direction_confidence_threshold: 0.5, person_confidence_threshold: 0.5) and only transitions when confidence sufficient.
- **Search timeout:** If person not found within search_timeout_s (default 10s), SEARCH_FOR_PERSON transitions back to IDLE.
- **Lost person during approach:** APPROACH_PERSON transitions back to SEARCH_FOR_PERSON if person_detected becomes false.
- **Parameter validation:** No explicit validation; relies on ROS 2 parameter loading and developer correct config files.

## Cross-Cutting Concerns

**Logging:**
- Approach: Direct use of rclpy node.get_logger() in each node
- Patterns: Info on state transitions, error/warning on timeout/lost detections, debug for each publish (not implemented)
- No centralized logging service

**Validation:**
- Approach: Type hints in Python, dataclass constraints via field definitions
- Input validation: Confidence bounded to [0.0, 1.0] in providers; behavior thresholds checked before transitions
- No schema validation on ROS messages (std_msgs are untyped Float64MultiArray until migration to come_here_msgs)

**Authentication:**
- Approach: Not applicable. No external APIs or services.
- ROS 2 network is assumed trustworthy (local or VPN)

---

*Architecture analysis: 2026-04-06*
