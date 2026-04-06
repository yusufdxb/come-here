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
