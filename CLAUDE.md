# come-here — Claude Code Instructions

## What This Is

ROS 2 workspace for Unitree GO2 "come here" command response. The robot hears "come here," estimates sound direction, rotates toward it, detects the speaker visually, and approaches them.

## Project Location

This project lives on a portable T7 SSD. The mount point varies by machine.
Run `pwd` to determine the current path before referencing files.

## Architecture

5 ROS 2 packages:

| Package | Role |
|---|---|
| `come_here_msgs` | Message definitions (AudioDirection, PersonDetection, WakePhrase) |
| `come_here_audio` | Audio direction estimation + wake phrase detection |
| `come_here_perception` | Visual person detection and tracking |
| `come_here_behavior` | State machine: IDLE -> LISTENING -> TURN_TO_SOUND -> SEARCH_FOR_PERSON -> APPROACH_PERSON -> STOP |
| `come_here_bringup` | Launch files and combined config |

## Key Abstractions

All sensor inputs are behind ABCs with mock implementations:

- **`AudioDirectionProvider`** (`come_here_audio/audio_direction_provider.py`) — sound source direction. Microphone hardware is unknown. Implement a new subclass when the mic is chosen.
- **`WakePhraseDetector`** (`come_here_audio/wake_phrase_detector.py`) — wake phrase detection. `WhisperPhraseDetector` is the real implementation (faster-whisper or HF+LoRA).
- **`PersonDetector`** (`come_here_perception/person_detector.py`) — visual person detection. Stubbed until camera integration.

## Whisper Setup

- Pre-cached models in `models/` (no internet needed):
  - `models/whisper-base.en/` — HuggingFace weights (for fine-tuning + LoRA inference)
  - `models/faster-whisper-base.en/` — CTranslate2 weights (for fast base inference)
- Code auto-resolves local model paths. No HuggingFace downloads required.
- All pip deps cached in `deps/` for offline install.

## Fine-Tuning Pipeline

In `training/`:
1. `record_samples.py` — record positive/negative audio samples
2. `finetune_whisper.py` — LoRA fine-tune on GPU (base.en = 0.15 GB VRAM)
3. `evaluate.py` — accuracy eval, base vs fine-tuned comparison, live mic test

## ROS Topics

- `/come_here/wake_phrase` — wake phrase events
- `/come_here/audio_direction` — sound source direction [azimuth, confidence]
- `/come_here/person_detection` — person detection [bearing, distance, confidence, detected]
- `/come_here/cmd_rotate` — rotation commands from behavior
- `/come_here/cmd_move` — forward motion commands from behavior
- `/come_here/state` — current state name
- `/come_here/mock_trigger` — simulate wake phrase (testing)
- `/come_here/mock_person` — simulate person detection (testing)

## Build & Run

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch come_here_bringup come_here.launch.py
```

## Rules

- **No fake completion.** Do not claim hardware works unless tested on hardware. Always separate verified / inferred / not yet validated.
- **Mic hardware is unknown.** Do not assume any specific mic. Implement new providers by subclassing `AudioDirectionProvider`.
- **Mock-first.** All sensors default to mock. Set `use_mock:=false` only after real providers exist.
- **Keep scope tight.** Do not touch `build/`, `install/`, `log/`, `models/`, `deps/`.
- **Nodes use std_msgs temporarily.** Migration to `come_here_msgs` is pending first colcon build.
- **Owner:** Yusuf Guenena (yusuf.a.guenena@gmail.com)

## What's Not Done Yet

- Real `AudioDirectionProvider` (blocked on mic hardware)
- Real `PersonDetector` (needs YOLO/MediaPipe + GO2 camera)
- GO2 locomotion bridge (`cmd_rotate`/`cmd_move` -> Unitree SDK or `cmd_vel`)
- Migration from `std_msgs` to `come_here_msgs`
- Whisper fine-tuning (pipeline ready, needs recorded samples)

<!-- GSD:project-start source:PROJECT.md -->
## Project

**come-here**

A ROS 2 system for the Unitree GO2 robot that responds to the voice command "come here." The robot hears the phrase via a ReSpeaker Mic Array v2.0, estimates the speaker's direction, rotates toward them, visually detects and locks on using YOLO on the GO2's camera, and walks to them. Runs entirely on a Jetson Orin NX mounted on the robot.

**Core Value:** The robot hears "come here" and physically comes to the speaker. Sound gets it facing the right direction, vision gets it to the right person.

### Constraints

- **Compute**: Jetson Orin NX — must run YOLO + Whisper concurrently within GPU/CPU budget
- **Network**: Jetson has no reliable DNS (hotspot blocks port 53) — pip installs must be scp'd from PC
- **Audio**: ReSpeaker is UAC1.0 at 16kHz — sounddevice must use the correct ALSA device
- **Latency**: DOA → rotation should feel responsive (<1s from voice to movement start)
- **Deployment**: All code must be deployable to Jetson via scp from T7 SSD
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3 - All ROS 2 packages and training scripts
- C++ - CMake build system for come_here_msgs (IDL generation)
- YAML - Configuration files for ROS 2 parameters
## Runtime
- ROS 2 Humble on Ubuntu 22.04 (from setup.sh line 62 and README)
- Python 3.10+ (inferred from wheel naming in deps/)
- pip - Python dependency installation
- colcon - ROS 2 build system and workspace tool
## Frameworks
- `rclpy` [Humble] - ROS 2 Python client library
- `ament_python` - Build tool for Python ROS 2 packages
- `ament_cmake` - Build tool for message definitions (come_here_msgs)
- `faster-whisper` [latest] - Fast Whisper inference via CTranslate2
- `transformers` [HuggingFace] - For Whisper with LoRA fine-tuning
- `peft` - Parameter-Efficient Fine-Tuning (LoRA support)
- `sounddevice` - Microphone audio input
- `torch` / `torchaudio` - PyTorch and audio utilities
- `accelerate` - Distributed training and inference optimization
- `datasets` [HuggingFace] - Audio dataset handling
- `numpy` - Numerical operations
- `soundfile` - WAV file reading/writing
## Key Dependencies
- `rclpy` - Enables ROS 2 node initialization and messaging
- `faster-whisper` - Enables wake phrase detection (CTranslate2 backend, GPU-optimized)
- `sounddevice` - Enables microphone audio acquisition
- `numpy` - Core signal processing for audio manipulation
- `transformers` - Enables LoRA fine-tuning path (optional but required if using adapter_path)
- `peft` - Enables fine-tuned LoRA adapter loading
- `torch` - Enables GPU acceleration (CUDA 12.8+ via PyTorch wheels)
- `datasets` - Enables audio dataset pipelines for training
- `accelerate` - Enables multi-GPU and mixed-precision training
- `soundfile` - Audio recording and fine-tuning scripts only
- `pytest` - Unit testing framework
- `pytest` dependency in come_here_audio, come_here_perception, come_here_behavior
## Configuration
- ROS 2 must be sourced: `source /opt/ros/humble/setup.bash`
- Workspace sourced after build: `source install/setup.bash`
- CUDA environment: setup.sh detects and uses CUDA 12.8+ if available
- `CMakeLists.txt` - come_here_msgs IDL compilation (rosidl_default_generators)
- `setup.py` - Python package metadata for audio, perception, behavior, bringup packages
- Launch files: `come_here_bringup/launch/come_here.launch.py` (ament_index for config file discovery)
- `come_here_audio/config/audio_params.yaml` - Wake detector type, Whisper model size, device (cpu/cuda), LoRA adapter path
- `come_here_perception/config/perception_params.yaml` - Mock mode flag, publish rate
- `come_here_behavior/config/behavior_params.yaml` - State machine thresholds and timeouts
## Platform Requirements
- Ubuntu 22.04 LTS
- ROS 2 Humble installation at `/opt/ros/humble/`
- Python 3.10+
- CUDA 12.8+ (optional, falls back to CPU per setup.sh line 41)
- GPU (tested on RTX 5070 with 12GB VRAM per finetune_whisper.py line 21)
- Jetson Orin NX or equivalent ARM64 compute
- ROS 2 Humble for ARM64
- Microphone hardware (not yet specified - AudioDirectionProvider is an ABC)
- Camera hardware (not yet specified - PersonDetector is an ABC)
- Network connectivity for model downloads or local model cache
- Local cache in `models/` directory:
- Fallback: Auto-downloads on first use if not cached
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Module files: `snake_case.py` (e.g., `audio_node.py`, `person_detector.py`, `wake_phrase_detector.py`)
- Abstract base classes: `abstraction_name.py` (e.g., `audio_direction_provider.py`, `person_detector.py`)
- Mock implementations: `mock_*.py` (e.g., `mock_audio_provider.py`)
- Specific implementations: `specific_name_*.py` (e.g., `whisper_phrase_detector.py`)
- Launch files: `module_name.launch.py` (e.g., `audio.launch.py`, `come_here.launch.py`)
- PascalCase for all classes (both abstract and concrete)
- Abstract base classes: descriptive names without "Abstract" prefix (e.g., `AudioDirectionProvider`, `WakePhraseDetector`, `PersonDetector`)
- Concrete implementations: descriptive with purpose (e.g., `MockAudioProvider`, `WhisperPhraseDetector`, `AudioNode`)
- ROS 2 nodes: suffix `Node` (e.g., `AudioNode`, `PerceptionNode`, `BehaviorNode`)
- snake_case for all function and method names
- Private methods: prefix with `_` (e.g., `_tick()`, `_wake_cb()`, `_setup()`)
- Callback methods: suffix with `_cb` (e.g., `_wake_cb()`, `_direction_cb()`, `_person_cb()`, `_mock_trigger_cb()`)
- Getter/setter pattern: direct naming without "get_" prefix when returning single value (e.g., `check()`, `detect()`, `get_direction()`)
- snake_case for all local and instance variables
- Private instance variables: prefix with `_` (e.g., `self._active`, `self._fixed_azimuth`, `self._state`)
- Constants: UPPER_CASE (not explicitly found in codebase but follows Python convention)
- Dataclass fields: descriptive snake_case with type hints (e.g., `azimuth_rad`, `confidence`, `bearing_rad`, `distance_m`)
- Use Python type hints throughout
- Optional[Type] or Type | None for nullable values (codebase uses union syntax `Type | None`)
- Dataclasses for simple data structures (see `DirectionEstimate`, `PersonEstimate`, `PhraseDetection`)
## Code Style
- No explicit linting config found (no `.flake8`, `.pylintrc`, or `pyproject.toml`)
- Line length: Appears to follow ~88 character limit (Black default)
- Indentation: 4 spaces
- Blank lines: 2 lines between top-level definitions, 1 line between methods
- Standard library imports first
- Third-party imports (rclpy, numpy, etc.) second
- Local/relative imports last
- No wildcard imports observed; all imports are explicit
- Guarded imports for optional dependencies: wrapped in try/except with `_AVAILABLE` flags (see `whisper_phrase_detector.py` lines 31-44)
## Import Organization
- No alias system detected (using fully qualified imports from package root)
- Import style: `from come_here_audio.module import ClassName`
## Code Structure and Patterns
- Every module starts with a docstring explaining purpose and usage
- ROS 2 nodes document their published/subscribed topics and parameters in docstring
- Abstractions document their interface and subclassing requirements
- Use `ABC` and `@abstractmethod` decorator (e.g., `AudioDirectionProvider`, `WakePhraseDetector`, `PersonDetector`)
- Document interface contract in docstring
- Include guidance for implementing subclasses in docstrings
- Inherit from `rclpy.node.Node`
- `__init__()` declares parameters, creates pub/sub, starts timer
- Use descriptive node name (lowercase, underscores, e.g., `'audio_node'`, `'perception_node'`)
- Implement `destroy_node()` override to cleanup resources (teardown providers)
- Main entry point pattern: `def main(args=None): ... finally: node.destroy_node(); rclpy.shutdown()`
- Subclass abstract interface
- Implement all abstract methods
- Add `setup()` and `teardown()` lifecycle methods for resource management
- For mock implementations, include settable state methods (e.g., `set_detected()`, `set_triggered()`)
- Use `@dataclass` for simple data structures
- Include docstring describing the class purpose
- Annotate all fields with types
- No methods (pure data carriers)
## Error Handling
- Raise `NotImplementedError` with descriptive message for unimplemented features (e.g., `audio_node.py` lines 55-58, `perception_node.py` lines 32-34)
- Import errors wrapped with informative messages (e.g., `whisper_phrase_detector.py` lines 86-94)
- Thread safety: exception swallowing with sleep retry in background threads (e.g., `whisper_phrase_detector.py` lines 194-198)
- Graceful shutdown: try/except KeyboardInterrupt in main() with finally block to cleanup
- Check for None before dereferencing (e.g., `behavior_node.py` lines 114-120: `if estimate is not None:`)
- Return None for "not found" or "no result" cases (not exceptions)
## Logging
- `self.get_logger().info()` for startup, state transitions, detections
- `self.get_logger().warn()` for timeouts, recoverable errors
- Log human-readable info with format strings (e.g., `f'Wake phrase detected: "{detection.phrase}" (confidence={detection.confidence:.2f})'`)
- Avoid logging in tight loops; log state changes only
## Comments
- Module docstrings (required on every file)
- Class docstrings for abstractions and complex classes (required)
- Method docstrings for public abstract methods (required)
- TODO comments for incomplete/unimplemented work (e.g., `audio_node.py` line 23, 48, 80; `behavior_node.py` line 124, 156)
- Inline comments for non-obvious logic (e.g., `whisper_phrase_detector.py` lines 113-119: explain local model resolution)
- Docstrings use triple-quoted format with description, then blank line, then details
- Parameter documentation: use ROS 2 node docstring format listing Publishes/Subscribes/Parameters
- Avoid redundant comments (code is self-documenting when naming is clear)
## Function Design
- Most methods 5-30 lines; state machine callbacks typically 3-10 lines
- Longer methods (40+ lines) found in `WhisperPhraseDetector._listen_loop()` due to threaded event loop complexity
- Use type hints on all parameters
- ROS 2 node constructors take no required args (`__init__(self)`)
- Provider initialization accepts configuration: `__init__(self, model_size: str = ..., device: str = ..., adapter_path: Optional[str] = None)`
- Callback signatures match ROS 2 pattern: `def callback(self, msg: MessageType)`
- Always annotated with type hints (including `-> None`)
- Return dataclass instances for structured results (e.g., `DirectionEstimate`, `PersonEstimate`, `PhraseDetection`)
- Return None for "no result" rather than raising exceptions
- Interface methods document return contract in docstring (e.g., "Returns PhraseDetection if detected, None otherwise")
## Module Design
- No explicit `__all__` found; all public classes are importable
- Mock implementations and abstractions co-located in same file when tightly coupled (e.g., `mock_audio_provider.py` in separate file but `mock_audio_provider.py` imports from `audio_direction_provider.py`)
- Each package's `__init__.py` is empty (no re-exports)
- Not used; clients import directly from module files (e.g., `from come_here_audio.audio_node import AudioNode`)
## ROS 2 Specific Conventions
- Use leading slash: `/come_here/wake_phrase`, `/come_here/audio_direction`
- Use snake_case for topic names
- Organize by subsystem prefix: `/come_here/`
- Declare with snake_case: `self.declare_parameter('use_mock', True)`
- Retrieve with same name: `self.get_parameter('use_mock').value`
- Document defaults and types in docstring
- Publish data packed into Float64MultiArray with documented element order (e.g., `behavior_node.py` line 14 documents `[azimuth, confidence]`)
- Unpack with length check and indexing (e.g., `if len(msg.data) >= 2: self._last_azimuth = msg.data[0]`)
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Three independent ROS 2 nodes (audio, perception, behavior) communicate via typed topics
- All sensor inputs (direction, wake phrase, person detection) abstracted behind ABC interfaces
- Behavior driven by a 6-state finite state machine triggered by sensor events
- Default-to-mock approach: all providers have mock implementations, real providers raise `NotImplementedError`
- Non-blocking sensor polling: background threads for audio, synchronous for vision
## Layers
- Purpose: Define strongly-typed ROS messages for cross-node communication
- Location: `come_here_msgs/msg/`
- Contains: Three message definitions (AudioDirection, PersonDetection, WakePhrase)
- Depends on: `std_msgs`
- Used by: All three functional nodes
- Purpose: Detect wake phrase and estimate sound source direction
- Location: `come_here_audio/come_here_audio/`
- Contains: Two provider abstractions (AudioDirectionProvider, WakePhraseDetector) and a ROS node
- Depends on: Audio providers (mock or real)
- Used by: Behavior node via `/come_here/wake_phrase` and `/come_here/audio_direction` topics
- Purpose: Detect and localize a person in the camera frame
- Location: `come_here_perception/come_here_perception/`
- Contains: PersonDetector abstraction and perception ROS node
- Depends on: Person detection provider (mock or real)
- Used by: Behavior node via `/come_here/person_detection` topic
- Purpose: State machine that orchestrates the "come here" sequence
- Location: `come_here_behavior/come_here_behavior/`
- Contains: State enum and BehaviorNode that transitions between IDLE, LISTENING, TURN_TO_SOUND, SEARCH_FOR_PERSON, APPROACH_PERSON, STOP
- Depends on: Wake phrase, audio direction, person detection subscriptions
- Used by: Robot locomotion layer (cmd_rotate, cmd_move topics)
- Purpose: Unified launch and configuration entry point
- Location: `come_here_bringup/launch/`
- Contains: come_here.launch.py that starts all three nodes with shared parameters
- Depends on: All functional packages
- Used by: End user to start the system
## Data Flow
- Audio perception: Thread-safe queue in WhisperPhraseDetector, pull-based via check()
- Visual perception: Mock detector holds state (_detected, _bearing, _distance), modified externally via set_detected()
- Behavior: All state in BehaviorNode (_state, _last_azimuth, _last_dir_confidence, _person_*, _search_start_time)
- No persistent state: System resets to IDLE at completion
## Key Abstractions
- Purpose: Decouple ROS node from specific microphone hardware
- Examples: `come_here_audio/audio_direction_provider.py` (ABC), `come_here_audio/mock_audio_provider.py` (mock)
- Pattern: Abstract base class with setup(), get_direction(), teardown(); returns DirectionEstimate dataclass (azimuth_rad, confidence)
- Blocking behavior: None, returns immediately with None if no sound
- Purpose: Decouple ROS node from specific ASR engine
- Examples: `come_here_audio/wake_phrase_detector.py` (ABC), `come_here_audio/whisper_phrase_detector.py` (real with HF + LoRA support)
- Pattern: Abstract base class with setup(), check(), teardown(); returns PhraseDetection dataclass (phrase, confidence) or None
- Concurrency model: WhisperPhraseDetector runs inference in background thread, check() is non-blocking via queue.Queue
- Purpose: Decouple ROS node from specific vision model
- Examples: `come_here_perception/person_detector.py` (ABC), stub real implementation pending YOLO/MediaPipe integration
- Pattern: Abstract base class with setup(), detect(), teardown(); returns PersonEstimate dataclass (bearing_rad, distance_m, confidence, detected)
- Blocking behavior: None, returns immediately
## Entry Points
- Location: `come_here_audio/come_here_audio/audio_node.py`
- Triggers: Launched by come_here.launch.py
- Responsibilities: Initialize AudioDirectionProvider and WakePhraseDetector based on parameters; tick at fixed rate to publish audio estimates; handle mock trigger subscriptions
- Location: `come_here_perception/come_here_perception/perception_node.py`
- Triggers: Launched by come_here.launch.py
- Responsibilities: Initialize PersonDetector; tick at fixed rate to publish person detection; handle mock person toggle subscriptions
- Location: `come_here_behavior/come_here_behavior/behavior_node.py`
- Triggers: Launched by come_here.launch.py
- Responsibilities: Subscribe to wake_phrase, audio_direction, person_detection; tick state machine; publish state, cmd_rotate, cmd_move
- Location: `come_here_bringup/launch/come_here.launch.py`
- Triggers: `ros2 launch come_here_bringup come_here.launch.py [use_mock:=true|false]`
- Responsibilities: Load config files from all three packages; instantiate and start all three nodes; forward use_mock parameter to each node's config
## Error Handling
- **Missing real providers:** BehaviorNode, AudioNode, PerceptionNode raise NotImplementedError immediately if real provider requested but not implemented. Forces developer to explicitly set use_mock:=true or implement provider.
- **Audio input failures:** WhisperPhraseDetector thread catches audio recording exceptions, logs, sleeps 0.5s, retries. Non-blocking: doesn't crash main thread.
- **Silent audio chunks:** WhisperPhraseDetector skips chunks with max amplitude < 0.01 to avoid processing pure silence.
- **Low-confidence estimates:** Behavior node respects confidence thresholds (direction_confidence_threshold: 0.5, person_confidence_threshold: 0.5) and only transitions when confidence sufficient.
- **Search timeout:** If person not found within search_timeout_s (default 10s), SEARCH_FOR_PERSON transitions back to IDLE.
- **Lost person during approach:** APPROACH_PERSON transitions back to SEARCH_FOR_PERSON if person_detected becomes false.
- **Parameter validation:** No explicit validation; relies on ROS 2 parameter loading and developer correct config files.
## Cross-Cutting Concerns
- Approach: Direct use of rclpy node.get_logger() in each node
- Patterns: Info on state transitions, error/warning on timeout/lost detections, debug for each publish (not implemented)
- No centralized logging service
- Approach: Type hints in Python, dataclass constraints via field definitions
- Input validation: Confidence bounded to [0.0, 1.0] in providers; behavior thresholds checked before transitions
- No schema validation on ROS messages (std_msgs are untyped Float64MultiArray until migration to come_here_msgs)
- Approach: Not applicable. No external APIs or services.
- ROS 2 network is assumed trustworthy (local or VPN)
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
