# Technology Stack

**Analysis Date:** 2026-04-06

## Languages

**Primary:**
- Python 3 - All ROS 2 packages and training scripts
- C++ - CMake build system for come_here_msgs (IDL generation)

**Secondary:**
- YAML - Configuration files for ROS 2 parameters

## Runtime

**Environment:**
- ROS 2 Humble on Ubuntu 22.04 (from setup.sh line 62 and README)
- Python 3.10+ (inferred from wheel naming in deps/)

**Package Manager:**
- pip - Python dependency installation
- colcon - ROS 2 build system and workspace tool

## Frameworks

**Core ROS 2:**
- `rclpy` [Humble] - ROS 2 Python client library
  - Used in: `come_here_audio.audio_node`, `come_here_perception.perception_node`, `come_here_behavior.behavior_node`
- `ament_python` - Build tool for Python ROS 2 packages
- `ament_cmake` - Build tool for message definitions (come_here_msgs)

**Audio & Speech:**
- `faster-whisper` [latest] - Fast Whisper inference via CTranslate2
  - Used in: `come_here_audio.whisper_phrase_detector` (line 33-34)
  - Backup backend when no LoRA adapter is specified
- `transformers` [HuggingFace] - For Whisper with LoRA fine-tuning
  - Used in: `come_here_audio.whisper_phrase_detector` (line 40-41)
  - Alternative backend when `adapter_path` is set
- `peft` - Parameter-Efficient Fine-Tuning (LoRA support)
  - Used in: `come_here_audio.whisper_phrase_detector` (line 41)
  - Loads fine-tuned LoRA adapters on top of base Whisper
- `sounddevice` - Microphone audio input
  - Used in: `come_here_audio.whisper_phrase_detector._listen_loop()` (line 166)
  - Records audio chunks from system default or specified device

**ML & Compute:**
- `torch` / `torchaudio` - PyTorch and audio utilities
  - Required for transformers backend and training
  - Installed with CUDA 12.8+ support per setup.sh line 24
- `accelerate` - Distributed training and inference optimization
  - Used in: training pipeline (`training/finetune_whisper.py`)
- `datasets` [HuggingFace] - Audio dataset handling
  - Used in: `training/finetune_whisper.py` (line 30)
  - Loads WAV files and casts to Audio column

**Scientific Computing:**
- `numpy` - Numerical operations
  - Used in: `come_here_audio.whisper_phrase_detector` for audio processing (line 27, 183)
  - Also in training scripts for signal processing

**Audio File I/O:**
- `soundfile` - WAV file reading/writing
  - Used in: `training/finetune_whisper.py` (line 41)

## Key Dependencies

**Critical:**
- `rclpy` - Enables ROS 2 node initialization and messaging
- `faster-whisper` - Enables wake phrase detection (CTranslate2 backend, GPU-optimized)
- `sounddevice` - Enables microphone audio acquisition
- `numpy` - Core signal processing for audio manipulation

**Infrastructure:**
- `transformers` - Enables LoRA fine-tuning path (optional but required if using adapter_path)
- `peft` - Enables fine-tuned LoRA adapter loading
- `torch` - Enables GPU acceleration (CUDA 12.8+ via PyTorch wheels)
- `datasets` - Enables audio dataset pipelines for training
- `accelerate` - Enables multi-GPU and mixed-precision training

**Optional (Development/Training):**
- `soundfile` - Audio recording and fine-tuning scripts only
- `pytest` - Unit testing framework
- `pytest` dependency in come_here_audio, come_here_perception, come_here_behavior

## Configuration

**Environment:**
- ROS 2 must be sourced: `source /opt/ros/humble/setup.bash`
- Workspace sourced after build: `source install/setup.bash`
- CUDA environment: setup.sh detects and uses CUDA 12.8+ if available

**Build:**
- `CMakeLists.txt` - come_here_msgs IDL compilation (rosidl_default_generators)
- `setup.py` - Python package metadata for audio, perception, behavior, bringup packages
  - Each declares entry points for console scripts (e.g., `audio_node` → `come_here_audio.audio_node:main`)
- Launch files: `come_here_bringup/launch/come_here.launch.py` (ament_index for config file discovery)

**Parameter Files (YAML):**
- `come_here_audio/config/audio_params.yaml` - Wake detector type, Whisper model size, device (cpu/cuda), LoRA adapter path
- `come_here_perception/config/perception_params.yaml` - Mock mode flag, publish rate
- `come_here_behavior/config/behavior_params.yaml` - State machine thresholds and timeouts

## Platform Requirements

**Development:**
- Ubuntu 22.04 LTS
- ROS 2 Humble installation at `/opt/ros/humble/`
- Python 3.10+
- CUDA 12.8+ (optional, falls back to CPU per setup.sh line 41)
- GPU (tested on RTX 5070 with 12GB VRAM per finetune_whisper.py line 21)

**Production (Unitree GO2):**
- Jetson Orin NX or equivalent ARM64 compute
- ROS 2 Humble for ARM64
- Microphone hardware (not yet specified - AudioDirectionProvider is an ABC)
- Camera hardware (not yet specified - PersonDetector is an ABC)
- Network connectivity for model downloads or local model cache

**Model Storage:**
- Local cache in `models/` directory:
  - `models/whisper-base.en/` - HF Whisper base.en (for transformers backend)
  - `models/faster-whisper-base.en/` - CTranslate2 Whisper base.en (for faster-whisper backend)
- Fallback: Auto-downloads on first use if not cached

---

*Stack analysis: 2026-04-06*
