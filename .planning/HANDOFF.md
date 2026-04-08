# Session Handoff — 2026-04-06

## What Was Achieved

### ReSpeaker Mic Array v2.0 — Fully Integrated
- `ReSpeakerDOAProvider` created at `come_here_audio/come_here_audio/respeaker_doa_provider.py`
- Reads DOAANGLE + VOICEACTIVITY registers via pyusb USB HID control transfers
- Converts degrees to ROS radians with configurable `frame_offset_deg` parameter
- Wired into `audio_node.py` — selected when `use_mock:=false`
- **Verified on hardware**: DOA fires when voice detected, silent when no voice

### Whisper Wake Phrase Detection — Working on Jetson
- faster-whisper base.en runs on Orin NX CPU with int8 quantization (~3s for 3s chunk)
- `WhisperPhraseDetector` updated: raw mic channel 1 with 4x gain boost, 6-channel recording
- `vad_filter=True` **crashes on ARM** (onnxruntime assertion) — disabled
- `no_speech_prob > 0.5` filter suppresses hallucinations from silence
- Model cached at `~/come-here/models/faster-whisper-base.en/.../snapshots/<hash>/`
- **Verified**: "come here" detected from live mic multiple times

### GO2 Locomotion — Working via Sport API
- Commands sent via `unitree_api/msg/Request` on `/api/sport/request` (DDS/CycloneDDS)
- API ID 1008 = `Move(x, y, z)` where z = yaw rotation rate
- API ID 1003 = StopMove, 1006 = RecoveryStand, 1002 = BalanceStand
- Need motion_switcher to "normal" mode first
- `error_code: 100` in sportmodestate is normal, doesn't block commands
- **Verified**: Robot rotated in place with Move(0, 0, 0.5)

### Voice Response — "I am coming" on GO2 Speaker
- Audio plays through GO2's audiohub API (NOT Jetson ALSA — Jetson has no speaker output to robot)
- Protocol: `/api/audiohub/request` with api_id 4001=start, 4003=chunk (base64 WAV), 4002=end
- Voice: edge-tts AriaNeural (natural US female), saved as `i_am_coming.wav`
- **Verified**: Robot spoke "I am coming" through its built-in speaker

### End-to-End Demo — Partially Working
- `come_here_audio/scripts/hear_and_rotate_demo.py` combines all components
- Successfully: heard "come here" → said "I am coming" → rotated toward speaker
- Issues: DOA frame offset not calibrated, ReSpeaker USB disconnected at end of session

### GSD Project Structure
- `.planning/` initialized with PROJECT.md, REQUIREMENTS.md, ROADMAP.md (4 phases, 19 reqs)
- Research docs in `.planning/research/` (STACK, FEATURES, ARCHITECTURE, PITFALLS, SUMMARY)
- Codebase map in `.planning/codebase/` (7 docs)

## What To Do Next

### Immediate Fixes
1. **ReSpeaker USB may need replug** — last test got "No input device matching hw:0,0"
2. **Set AGC on startup** — AGCMAXGAIN=100, GAMMAVAD_SR=3.5 (resets on USB replug, should be set in ReSpeakerDOAProvider.setup())
3. **Calibrate DOA frame_offset_deg** — mic 0 orientation vs robot forward. Spin robot, note which angle = forward, set offset

### Phase 2 Remaining (Audio)
- **WAKE-03**: Ring buffer refactor — current listen loop blocks 3s during recording + 3s during inference = ~6s cycle with gaps
- **WAKE-04**: Motor noise validation — no_speech_prob filter is coded, needs testing with robot moving

### Phase 3 (Vision) — Not Started
- Camera topic is `/frontvideostream` type `unitree_go/msg/Go2FrontVideoData` (NOT sensor_msgs/Image)
- `go2_ros2_sdk` at `~/go2_ws/src/go2_ros2_sdk/` has camera decoding code in `ros2_publisher.py`
- Need: install ultralytics, export YOLO11n to TensorRT FP16 on Jetson, implement YoloPersonDetector
- Person bearing from bbox center + HFOV, distance from bbox height, stop at ~0.8m

### Phase 4 (Integration) — Not Started
- Migrate std_msgs → come_here_msgs
- Full state machine: hear → speak → rotate → detect person → approach → stop
- GPU memory check with tegrastats (YOLO on GPU + Whisper on CPU should be fine)

## Key Environment Info

- **SSH**: `sshpass -p '123' ssh unitree@172.20.10.6` (WiFi hotspot, no ethernet needed)
- **ROS source**: `source /opt/ros/humble/setup.bash && source ~/go2_ws/install/setup.bash`
- **DDS env**: `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && export CYCLONEDDS_URI=file:///home/unitree/unitree_ros2/cyclonedds_ws/src/cyclonedds.xml`
- **No DNS on Jetson** — pip packages must be downloaded on PC and scp'd
- **GO2**: reachable at 192.168.123.161 from Jetson ethernet
- **T7 SSD**: project at `/media/careslab/T7 Storage/come-here/`

## Critical Findings (Don't Re-learn These)
- Use **raw mic channel 1** for Whisper, not beamformed ch0 (too quiet at distance)
- Apply **4x gain boost** before feeding to Whisper
- **vad_filter=True crashes** on ARM Jetson — use ReSpeaker hardware VAD instead
- GO2 speaker is via **audiohub DDS API**, not ALSA from Jetson
- Sport commands need **motion_switcher to "normal"** mode to work
- CTranslate2 + faster-whisper **do work** on this Jetson (ARM build issue didn't hit us)
