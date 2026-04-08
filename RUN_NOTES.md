# Come-Here Demo — Run Notes

## Prerequisites
- GO2 robot powered on and standing
- ReSpeaker Mic Array v2.0 plugged into Jetson USB
- Jetson connected via WiFi hotspot (SSH: `sshpass -p '123' ssh unitree@172.20.10.6`)

## 1. Set ReSpeaker AGC (resets on replug)

```bash
cd ~/usb_4_mic_array && python3 -c "
import sys; sys.path.insert(0, '.')
from tuning import Tuning
import usb.core
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
t = Tuning(dev)
t.write('AGCMAXGAIN', 1000.0)
t.write('AGCONOFF', 1)
t.write('GAMMAVAD_SR', 5.0)
print('AGC:', t.read('AGCMAXGAIN'), 'VAD:', t.read('GAMMAVAD_SR'))
"
```

## 2. Source ROS2 environment

```bash
source /opt/ros/humble/setup.bash
source ~/go2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/unitree/unitree_ros2/cyclonedds_ws/src/cyclonedds.xml
```

## 3. Run the demo

```bash
cd ~/come-here
python3 -u come_here_audio/scripts/hear_and_rotate_demo.py
```

Say "come here" — robot says "I am coming" and rotates toward you.

## Calibration

- DOA calibrated 2026-04-07: `frame_offset_deg = 29.0`
- If mic is remounted, re-run: `python3 calibrate_doa.py` (stand in front, speak, Ctrl+C)

## Known Issues

- ~3-6s latency (3s recording + ~3s CPU inference) — fix: build CTranslate2 with CUDA (see TODO_ctranslate2_cuda.md)
- Beamformed ch0 suppresses too much — using raw ch1 with 10x software gain
- Must be within ~1-2m for reliable detection with current gain settings
- `timeout` exit crash at end is cosmetic (ROS context destroyed after SIGTERM)

## Files

| File | Location on Jetson |
|------|--------------------|
| Demo script | `~/come-here/come_here_audio/scripts/hear_and_rotate_demo.py` |
| DOA provider | `~/come-here/come_here_audio/come_here_audio/respeaker_doa_provider.py` |
| Calibration | `~/come-here/come_here_audio/scripts/calibrate_doa.py` |
| Clap test | `~/come-here/come_here_audio/scripts/clap_and_rotate.py` |
| Voice WAV | `~/come-here/i_am_coming.wav` |
| Whisper model | `~/come-here/models/faster-whisper-base.en/...` |
| ReSpeaker SDK | `~/usb_4_mic_array/tuning.py` |
