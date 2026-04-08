# come-here

Audio-visual approach system for the Unitree GO2. The robot hears "come here," estimates the sound direction, rotates toward it, detects the speaker visually, and walks to them.

## Architecture

```
                    +-----------+
                    |  Behavior |
                    |  (state   |
                    |  machine) |
                    +-----+-----+
                     /    |    \
                    v     v     v
            cmd_rotate  cmd_move  state
                    ^     ^
                    |     |
        +-----------+   +-------------+
        |   Audio   |   | Perception  |
        |   Node    |   |   Node      |
        +-----------+   +-------------+
        wake_phrase     person_detection
        audio_direction
```

### Packages

| Package | Purpose |
|---------|---------|
| `come_here_msgs` | Message definitions (AudioDirection, PersonDetection, WakePhrase) |
| `come_here_audio` | Audio direction estimation + wake phrase detection |
| `come_here_perception` | Visual person detection and tracking |
| `come_here_behavior` | State machine: IDLE -> LISTENING -> TURN_TO_SOUND -> SEARCH_FOR_PERSON -> APPROACH_PERSON -> STOP |
| `come_here_bringup` | Launch files and combined configuration |

### State Machine

```
IDLE  --(wake phrase)--> LISTENING
LISTENING --(direction confident)--> TURN_TO_SOUND
TURN_TO_SOUND --(rotation issued)--> SEARCH_FOR_PERSON
SEARCH_FOR_PERSON --(person found)--> APPROACH_PERSON
SEARCH_FOR_PERSON --(timeout)--> IDLE
APPROACH_PERSON --(close enough)--> STOP
APPROACH_PERSON --(person lost)--> SEARCH_FOR_PERSON
STOP --> IDLE
```

### Key Abstractions

- **`AudioDirectionProvider`** (`come_here_audio/audio_direction_provider.py`): ABC for sound source direction estimation. Swap in real mic hardware by subclassing this.
- **`WakePhraseDetector`** (`come_here_audio/wake_phrase_detector.py`): ABC for wake phrase detection.
  - **`WhisperPhraseDetector`** (`whisper_phrase_detector.py`): Real implementation using faster-whisper. Runs Whisper in a background thread on rolling audio chunks, triggers on "come here" in the transcript.
- **`PersonDetector`** (`come_here_perception/person_detector.py`): ABC for visual person detection. Plug in YOLO, MediaPipe, etc.

All three have mock implementations for local development.

## Setup

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Build
cd /path/to/come-here
colcon build --symlink-install

# Source workspace
source install/setup.bash

# Launch (mock mode)
ros2 launch come_here_bringup come_here.launch.py use_mock:=true

# For Whisper wake phrase detection (requires faster-whisper + sounddevice):
# pip install faster-whisper sounddevice numpy
# Then set wake_detector param to 'whisper' in audio_params.yaml
```

### Testing Mock Trigger

```bash
# Simulate "come here" detection
ros2 topic pub --once /come_here/mock_trigger std_msgs/Bool "data: true"

# Simulate person appearing
ros2 topic pub --once /come_here/mock_person std_msgs/Bool "data: true"

# Watch state transitions
ros2 topic echo /come_here/state
```

### Unit Tests

```bash
cd come_here_audio && python -m pytest test/
cd come_here_perception && python -m pytest test/
cd come_here_behavior && python -m pytest test/
```

## Assumptions

- ROS 2 Humble on Ubuntu 22.04
- Unitree GO2 robot (locomotion commands are stubs)
- Microphone hardware is **unknown** -- the `AudioDirectionProvider` interface exists but no real implementation yet
- Camera assumed available on the GO2 but perception node is stubbed

## Hardware-Dependent Work (Not Yet Implemented)

1. **Microphone array integration**: Need mic model to implement a real `AudioDirectionProvider` (GCC-PHAT, MUSIC, SRP-PHAT, etc.)
2. **Wake phrase validation**: `WhisperPhraseDetector` is implemented but not yet tested on hardware. Needs validation with the actual mic and ambient noise conditions
3. **Person detection model**: Need to integrate YOLO or equivalent with the GO2's camera and implement a real `PersonDetector`
4. **GO2 locomotion bridge**: `cmd_rotate` and `cmd_move` topics are placeholders -- need to bridge to the Unitree SDK or `cmd_vel`
5. **Message migration**: Switch from `std_msgs` placeholders to `come_here_msgs` custom messages after first `colcon build`

## Next Steps When Microphone is Provided

1. Identify the mic's driver/SDK (e.g., ReSpeaker USB, XMOS, Matrix Voice)
2. Create `come_here_audio/come_here_audio/real_audio_provider.py` subclassing `AudioDirectionProvider`
3. Implement `setup()` with device init, `get_direction()` with the mic's DoA algorithm, `teardown()` with cleanup
4. Add the provider to the `audio_node.py` selection logic (the `else` branch)
5. Test with `use_mock:=false`
6. Choose and integrate a wake phrase engine that works with the same mic input
