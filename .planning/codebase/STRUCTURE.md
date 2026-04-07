# Codebase Structure

**Analysis Date:** 2026-04-06

## Directory Layout

```
come-here/                            # ROS 2 workspace root
├── come_here_msgs/                   # Message definitions (C++ CMake build)
│   ├── msg/                          # .msg files
│   ├── package.xml
│   └── CMakeLists.txt
├── come_here_audio/                  # Audio perception node (Python)
│   ├── come_here_audio/              # Package source
│   │   ├── audio_node.py             # ROS node entry point
│   │   ├── audio_direction_provider.py  # Direction estimation ABC
│   │   ├── mock_audio_provider.py    # Mock direction provider
│   │   ├── wake_phrase_detector.py   # Wake phrase detection ABC
│   │   ├── whisper_phrase_detector.py   # Whisper-based detector (real)
│   │   └── __init__.py
│   ├── launch/
│   │   └── audio.launch.py
│   ├── config/
│   │   └── audio_params.yaml
│   ├── test/
│   │   ├── test_audio_provider.py
│   │   └── __init__.py
│   ├── package.xml
│   └── setup.py
├── come_here_perception/             # Visual perception node (Python)
│   ├── come_here_perception/         # Package source
│   │   ├── perception_node.py        # ROS node entry point
│   │   ├── person_detector.py        # Person detection ABC and mock
│   │   └── __init__.py
│   ├── config/
│   │   └── perception_params.yaml
│   ├── test/
│   │   ├── test_person_detector.py
│   │   └── __init__.py
│   ├── package.xml
│   └── setup.py
├── come_here_behavior/               # Behavior state machine (Python)
│   ├── come_here_behavior/           # Package source
│   │   ├── behavior_node.py          # ROS node + state machine
│   │   └── __init__.py
│   ├── config/
│   │   └── behavior_params.yaml
│   ├── test/
│   │   ├── test_state_machine.py
│   │   └── __init__.py
│   ├── package.xml
│   └── setup.py
├── come_here_bringup/                # System launch and config
│   ├── come_here_bringup/
│   │   └── __init__.py
│   ├── launch/
│   │   └── come_here.launch.py       # Main launch file
│   ├── config/
│   │   └── come_here_params.yaml     # (Optional shared params)
│   ├── package.xml
│   └── setup.py
├── training/                         # Whisper fine-tuning pipeline
│   ├── record_samples.py             # Record positive/negative audio samples
│   ├── finetune_whisper.py           # LoRA fine-tuning on GPU
│   ├── evaluate.py                   # Accuracy eval and live testing
│   ├── data/                         # Recorded training samples (empty by default)
│   └── README.md
├── models/                           # Pre-cached model weights (no internet needed)
│   ├── whisper-base.en/              # HuggingFace format (for fine-tuning)
│   └── faster-whisper-base.en/       # CTranslate2 format (for fast inference)
├── deps/                             # Pip packages cached for offline install
├── .planning/                        # GSD planning documents
│   └── codebase/
├── README.md                         # Project overview and assumptions
├── CLAUDE.md                         # Claude Code instructions
├── AGENTS.md                         # Agent working norms
└── setup.sh                          # Workspace setup script
```

## Directory Purposes

**come_here_msgs:**
- Purpose: ROS 2 interface definitions
- Contains: Three custom message types (AudioDirection, PersonDetection, WakePhrase)
- Key files: `msg/AudioDirection.msg`, `msg/PersonDetection.msg`, `msg/WakePhrase.msg`
- Build type: ament_cmake (C++ build)
- Generates: Python bindings via rosidl_default_generators

**come_here_audio:**
- Purpose: Audio perception: direction estimation and wake phrase detection
- Contains: Provider ABCs, mock implementations, Whisper-based detector, ROS node
- Key files: `come_here_audio/audio_node.py`, `audio_direction_provider.py`, `whisper_phrase_detector.py`
- Configuration: `config/audio_params.yaml` (use_mock, wake_detector type, Whisper model size, device)
- Tests: `test/test_audio_provider.py` (provider and mock detector tests)

**come_here_perception:**
- Purpose: Visual perception: person detection and localization
- Contains: PersonDetector ABC, mock implementation, ROS node
- Key files: `come_here_perception/perception_node.py`, `person_detector.py`
- Configuration: `config/perception_params.yaml` (use_mock, publish rate)
- Tests: `test/test_person_detector.py` (detector and mock tests)

**come_here_behavior:**
- Purpose: Behavior orchestration: state machine logic
- Contains: State enum, BehaviorNode with all transitions
- Key files: `come_here_behavior/behavior_node.py` (combined state machine + node)
- Configuration: `config/behavior_params.yaml` (thresholds, timeouts, speeds)
- Tests: `test/test_state_machine.py` (state enum existence only)

**come_here_bringup:**
- Purpose: System-level launch and coordination
- Contains: Main launch file, optional shared config
- Key files: `launch/come_here.launch.py` (declares use_mock arg, instantiates three nodes with config)
- Configuration: Loads audio_params.yaml, perception_params.yaml, behavior_params.yaml from respective packages

**training/:**
- Purpose: Whisper fine-tuning pipeline for wake phrase accuracy improvement
- Contains: Data collection, LoRA fine-tuning, evaluation scripts
- Key files: `record_samples.py`, `finetune_whisper.py`, `evaluate.py`
- Data: `data/` directory for recorded audio samples (empty until user adds samples)

**models/:**
- Purpose: Pre-cached model weights for offline use
- Contains: Two Whisper variants (HuggingFace for fine-tuning, CTranslate2 for fast inference)
- Not committed to git (cached locally on T7 SSD)
- Usage: WhisperPhraseDetector auto-resolves from `models/` first, falls back to HuggingFace download

**deps/:**
- Purpose: Offline pip dependency cache
- Not committed to git
- Usage: For systems without internet access

## Key File Locations

**Entry Points:**

- `come_here_bringup/launch/come_here.launch.py`: Start entire system. Usage: `ros2 launch come_here_bringup come_here.launch.py [use_mock:=true|false]`
- `come_here_audio/come_here_audio/audio_node.py`: Audio perception node. Executable: `audio_node` (configured in setup.py)
- `come_here_perception/come_here_perception/perception_node.py`: Visual perception node. Executable: `perception_node` (configured in setup.py)
- `come_here_behavior/come_here_behavior/behavior_node.py`: Behavior state machine. Executable: `behavior_node` (configured in setup.py)

**Configuration:**

- `come_here_audio/config/audio_params.yaml`: Audio node parameters (use_mock, wake_detector type, Whisper config)
- `come_here_perception/config/perception_params.yaml`: Perception node parameters (use_mock, publish rate)
- `come_here_behavior/config/behavior_params.yaml`: Behavior parameters (confidence thresholds, timeouts, approach speed)
- `come_here_bringup/config/come_here_params.yaml`: Optional shared parameters (unused by default)

**Core Logic:**

- `come_here_audio/come_here_audio/audio_direction_provider.py`: Direction estimation ABC (interface for real mic implementations)
- `come_here_audio/come_here_audio/mock_audio_provider.py`: Mock direction provider (synthetic estimates for testing)
- `come_here_audio/come_here_audio/wake_phrase_detector.py`: Wake phrase detection ABC and MockWakePhraseDetector
- `come_here_audio/come_here_audio/whisper_phrase_detector.py`: Real Whisper-based detector (faster-whisper or HuggingFace with LoRA)
- `come_here_perception/come_here_perception/person_detector.py`: Person detection ABC and MockPersonDetector
- `come_here_behavior/come_here_behavior/behavior_node.py`: State machine (6 states, transitions on sensor events)

**Messages:**

- `come_here_msgs/msg/AudioDirection.msg`: azimuth_rad (float64), confidence (float64)
- `come_here_msgs/msg/PersonDetection.msg`: bearing_rad (float64), distance_m (float64), confidence (float64), detected (bool)
- `come_here_msgs/msg/WakePhrase.msg`: phrase (string), confidence (float64)

**Testing:**

- `come_here_audio/test/test_audio_provider.py`: Tests for MockAudioProvider and MockWakePhraseDetector
- `come_here_perception/test/test_person_detector.py`: Tests for MockPersonDetector
- `come_here_behavior/test/test_state_machine.py`: Tests for State enum
- **Run:** `cd <package> && python -m pytest test/`

## Naming Conventions

**Files:**

- Python source: `snake_case.py` (e.g., `audio_node.py`, `mock_audio_provider.py`)
- Config: `snake_case.yaml` (e.g., `audio_params.yaml`)
- Launch: `snake_case.launch.py` (e.g., `come_here.launch.py`)
- Tests: `test_snake_case.py` (e.g., `test_audio_provider.py`)
- Messages: `PascalCase.msg` (e.g., `AudioDirection.msg`, `WakePhrase.msg`)

**Directories:**

- ROS 2 packages: `snake_case` (e.g., `come_here_audio`, `come_here_behavior`)
- Python module source: same as package name (e.g., `come_here_audio/come_here_audio/`)
- Config: `config/` (lowercase)
- Tests: `test/` (lowercase)
- Launch: `launch/` (lowercase)

**Python Classes:**

- Abstract base classes: `PascalCase` with "ABC" or concept name (e.g., `AudioDirectionProvider`, `WakePhraseDetector`)
- Mock implementations: `MockPascalCase` (e.g., `MockAudioProvider`, `MockWakePhraseDetector`)
- Dataclasses: `PascalCase` for type names (e.g., `DirectionEstimate`, `PhraseDetection`, `PersonEstimate`)
- ROS nodes: `PascalCase` ending in "Node" (e.g., `AudioNode`, `PerceptionNode`, `BehaviorNode`)
- State enum: `State` with capitalized members (e.g., `State.IDLE`, `State.LISTENING`)

**ROS Topics:**

- Namespaced under `/come_here/` (e.g., `/come_here/wake_phrase`, `/come_here/audio_direction`)
- snake_case topic names (e.g., `/come_here/person_detection`, `/come_here/cmd_rotate`)

## Where to Add New Code

**New Wake Phrase Detector:**
- Implement a subclass of `WakePhraseDetector` in `come_here_audio/come_here_audio/custom_detector.py`
- Override `setup()`, `check()`, `teardown()`
- Register in `audio_node.py` as a new branch in the detector selection logic (after line 61)
- Update `audio_params.yaml` to add a new `wake_detector: custom_name` option

**New Audio Direction Provider:**
- Implement a subclass of `AudioDirectionProvider` in `come_here_audio/come_here_audio/real_audio_provider.py` (or hardware-specific name)
- Override `setup()`, `get_direction()`, `teardown()`
- Register in `audio_node.py` as a new branch in the provider selection logic (after line 49)
- Update `audio_params.yaml` to add a new configuration if needed

**New Person Detector:**
- Implement a subclass of `PersonDetector` in `come_here_perception/come_here_perception/yolo_detector.py` (or model name)
- Override `setup()`, `detect()`, `teardown()`
- Register in `perception_node.py` in the detector selection logic (after line 28)
- Update `perception_params.yaml` to add new options if needed

**New Behavior State:**
- Add new member to `State` enum in `come_here_behavior/come_here_behavior/behavior_node.py`
- Add transition logic in `_tick()` method (add elif block before the existing elif/else chain)
- Add test case to `test_state_machine.py`

**New Test:**
- File location: `<package>/test/test_module_name.py`
- Run with: `cd <package> && python -m pytest test/`
- Pattern: Use pytest (no special ROS test framework configured)

**New Configuration Parameter:**
- Add to relevant config file: `come_here_audio/config/audio_params.yaml`, `come_here_perception/config/perception_params.yaml`, or `come_here_behavior/config/behavior_params.yaml`
- Declare in node with `self.declare_parameter('param_name', default_value)`
- Access with `self.get_parameter('param_name').value`

**New ROS Topic:**
- Add publisher or subscriber in the relevant node
- Topic must follow `/come_here/topic_name` convention
- Use typed messages from `come_here_msgs` (or std_msgs temporarily)
- Document in node docstring

## Special Directories

**build/:**
- Purpose: Generated during `colcon build`
- Generated: Yes
- Committed: No (.gitignore)
- Do not touch: Contains intermediate build artifacts

**install/:**
- Purpose: Generated during `colcon build`
- Generated: Yes
- Committed: No (.gitignore)
- Contains: Installed packages, entry point scripts
- Do not touch: Regenerated on each build

**log/:**
- Purpose: Generated during ROS 2 launches
- Generated: Yes
- Committed: No (.gitignore)
- Contains: Launch logs, node output
- Safe to delete

**.git/:**
- Purpose: Git repository metadata
- Generated: Yes (by git init)
- Committed: N/A
- Do not touch

**models/:**
- Purpose: Pre-cached Whisper model weights
- Generated: No (downloaded once, cached locally)
- Committed: No (.gitignore)
- Cache strategy: HuggingFace Hub Cache with symlink from `models/whisper-base.en/` and `models/faster-whisper-base.en/`
- Offline use: WhisperPhraseDetector checks for local path before downloading

**deps/:**
- Purpose: Offline pip dependency cache
- Generated: No (user-created for offline installs)
- Committed: No (.gitignore)
- Usage: `pip install --no-index --find-links=deps -r requirements.txt` (not currently used, included for future)

---

*Structure analysis: 2026-04-06*
