# External Integrations

**Analysis Date:** 2026-04-06

## APIs & External Services

**Speech-to-Text:**
- OpenAI Whisper (via HuggingFace Model Hub)
  - SDK: `transformers` (HF backend) or `faster-whisper` (CTranslate2 backend)
  - Auth: None (models hosted on HuggingFace Hub, downloaded publicly)
  - Location: `come_here_audio.whisper_phrase_detector` (lines 32-44 guarded imports)
  - Model variants: base.en, small.en, medium.en (configurable per audio_params.yaml)
  - Compute: CTranslate2 (faster-whisper) preferred for inference; transformers required for LoRA fine-tuning

**Model Hub:**
- HuggingFace Model Hub (`https://huggingface.co/openai/`)
  - Auto-downloads Whisper models on first run if not cached locally
  - Models cached in `models/whisper-base.en/` and `models/faster-whisper-base.en/`
  - Supports local model paths as fallback per `whisper_phrase_detector._resolve_local_model()` (lines 113-119)

## Hardware Interfaces

**Microphone Input:**
- Framework: `sounddevice` library
- Location: `come_here_audio.whisper_phrase_detector._listen_loop()` (line 166)
- Sampling: 16kHz, mono, 2-second rolling chunks (configurable via whisper_chunk_duration_s)
- Device selection: System default or configurable via `mic_device` parameter
- Status: Currently unsupported - no real AudioDirectionProvider implementation exists (README line 104)
  - TODO: Identify mic hardware model and implement real DoA (Direction of Arrival) algorithm
  - Candidates: GCC-PHAT, MUSIC, SRP-PHAT per README line 109
  - Mock implementation available via `MockAudioProvider` for development

**Camera Input:**
- Status: Stubbed - no real PersonDetector implementation
- Location: `come_here_perception.perception_node` (line 32)
- Expected source: Unitree GO2 onboard camera
- TODO: Integrate YOLO or MediaPipe with GO2 camera per README lines 111-112
- Mock implementation available via `MockPersonDetector` for development

**Robot Locomotion:**
- GO2 Motion Commands: Published as `Float64` placeholders
  - `/come_here/cmd_rotate` - Rotation command (radians, placeholder)
  - `/come_here/cmd_move` - Forward velocity command (m/s, placeholder)
- Location: `come_here_behavior.behavior_node` (lines 78-79, 125-164)
- Status: Not yet integrated - needs bridge to Unitree SDK or `cmd_vel`
- TODO per README line 112: Replace placeholder topics with real GO2 motion bridge

## Data Storage

**Databases:**
- None - stateless ROS 2 nodes only

**File Storage:**
- Local filesystem only
  - Training data: `training/data/positive/` and `training/data/negative/` (WAV files)
  - Fine-tuned adapters: `training/output/lora_adapter/` (LoRA checkpoint)
  - Model cache: `models/whisper-base.en/` and `models/faster-whisper-base.en/` (auto-downloaded)

**Caching:**
- HuggingFace model caching: Automatic via `transformers` library (defaults to ~/.cache/huggingface/hub/)
- Local override: Project `models/` directory checked first via `_resolve_local_model()` (lines 113-119)
- faster-whisper cache: CTranslate2 format models in `models/faster-whisper-{model_size}/`

## Authentication & Identity

**Auth Provider:**
- None - public APIs only
  - HuggingFace models downloaded without credentials
  - All microphone and camera inputs are local

**Secrets/Credentials:**
- None required - no external service authentication needed

## Monitoring & Observability

**Error Tracking:**
- None detected - no external error reporting service integrated

**Logs:**
- ROS 2 logging via `self.get_logger()`
  - Location: All three nodes use ROS logger for info/warn output
  - Examples: `come_here_audio.audio_node:98`, `come_here_behavior.behavior_node:174`
- Stdout: Launch file sets `output='screen'` for all nodes (come_here_bringup/launch/come_here.launch.py lines 35, 41, 50)
- Exception handling: Graceful degradation in `whisper_phrase_detector._listen_loop()` (lines 194-198) catches transient audio errors

**Metrics:**
- No metrics infrastructure (Prometheus, CloudWatch, etc.)
- Manual inspection via ROS topic echo: `ros2 topic echo /come_here/{state,audio_direction,person_detection}` per README lines 88-89

## CI/CD & Deployment

**Hosting:**
- Jetson Orin NX (embedded in Unitree GO2 robot)
- No cloud deployment - edge inference only

**CI Pipeline:**
- None detected - no GitHub Actions, GitLab CI, Jenkins, etc. in .planning/ or root

**Build & Test:**
- `colcon build --symlink-install` - ROS 2 workspace build (README line 66)
- Unit tests: pytest in each package (README lines 95-97)
  - `come_here_audio/test/test_audio_provider.py`
  - `come_here_perception/test/test_person_detector.py`
  - `come_here_behavior/test/test_state_machine.py`
- No integration tests detected

## Environment Configuration

**Required env vars:**
- None explicit - ROS 2 environment sourced via setup.sh (line 59-69)
- CUDA environment: Detected automatically; fallback to CPU if not available

**ROS 2 Parameters (YAML-based):**
- Audio node parameters: `come_here_audio/config/audio_params.yaml`
  - `use_mock` - Boolean for mock provider
  - `wake_detector` - 'mock' or 'whisper'
  - `whisper_model_size` - 'tiny.en', 'base.en', 'small.en', 'medium.en'
  - `whisper_device` - 'cpu' or 'cuda'
  - `whisper_chunk_duration_s` - Audio chunk window (default 2.0)
  - `whisper_adapter_path` - Path to LoRA fine-tuned adapter (optional)

- Perception node parameters: `come_here_perception/config/perception_params.yaml`
  - `use_mock` - Boolean for mock detector
  - `publish_rate_hz` - Publishing frequency (default 10.0)

- Behavior node parameters: `come_here_behavior/config/behavior_params.yaml`
  - `direction_confidence_threshold` - Min confidence for direction (default 0.5)
  - `person_confidence_threshold` - Min confidence for person detection (default 0.5)
  - `approach_stop_distance_m` - Stopping distance (default 0.8m)
  - `search_timeout_s` - Max time searching for person (default 10s)
  - `approach_speed` - Forward velocity in approach state (default 0.3 m/s)

**Secrets location:**
- None - no secrets required

## ROS 2 Topics (Inter-Node Communication)

**Audio Node Outputs:**
- `/come_here/audio_direction` (std_msgs/Float64MultiArray) - [azimuth_rad, confidence]
- `/come_here/wake_phrase` (std_msgs/String) - Detected phrase ("come here" or "come over here")

**Audio Node Inputs (Mock):**
- `/come_here/mock_trigger` (std_msgs/Bool) - Simulate wake phrase detection (development only)

**Perception Node Outputs:**
- `/come_here/person_detection` (std_msgs/Float64MultiArray) - [bearing_rad, distance_m, confidence, detected]

**Perception Node Inputs (Mock):**
- `/come_here/mock_person` (std_msgs/Bool) - Simulate person detection (development only)

**Behavior Node Outputs:**
- `/come_here/cmd_rotate` (std_msgs/Float64) - Target rotation (radians)
- `/come_here/cmd_move` (std_msgs/Float64) - Forward velocity (m/s)
- `/come_here/state` (std_msgs/String) - Current state machine state

**Behavior Node Inputs:**
- `/come_here/wake_phrase` (std_msgs/String) - From audio node
- `/come_here/audio_direction` (std_msgs/Float64MultiArray) - From audio node
- `/come_here/person_detection` (std_msgs/Float64MultiArray) - From perception node

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- None detected

**Internal ROS 2 Callbacks:**
- `come_here_audio.audio_node._mock_trigger_cb()` - Subscribe handler for `/come_here/mock_trigger`
- `come_here_perception.perception_node._mock_person_cb()` - Subscribe handler for `/come_here/mock_person`
- `come_here_behavior.behavior_node._wake_cb()` - Subscribe handler for wake phrase
- `come_here_behavior.behavior_node._direction_cb()` - Subscribe handler for audio direction
- `come_here_behavior.behavior_node._person_cb()` - Subscribe handler for person detection

## Training & Fine-Tuning Pipeline

**Data Acquisition:**
- Script: `training/record_samples.py`
- Outputs WAV files to `training/data/positive/` and `training/data/negative/`
- Records via sounddevice

**Fine-Tuning:**
- Script: `training/finetune_whisper.py`
- Inputs: `training/data/positive/` (positive samples labeled "come here") and `training/data/negative/` (hard negatives)
- Framework: HuggingFace transformers + PEFT (LoRA)
- Output: LoRA adapter in `training/output/lora_adapter/`
- Deployment: Adapter path specified in `audio_params.yaml` → loaded by `whisper_phrase_detector` at setup()

**Evaluation:**
- Script: `training/evaluate.py`
- Supports live testing with fine-tuned adapter
- Validates wake phrase detection accuracy

---

*Integration audit: 2026-04-06*
