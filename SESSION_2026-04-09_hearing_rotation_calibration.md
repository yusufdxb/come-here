# Session: Hearing & Rotation Pipeline Calibration

## Session Date
2026-04-09

## Objective
Refine the hearing/audio localization pipeline so "come here" detection is more
accurate, and calibrate rotation so the robot turns to the correct direction.

## Work Completed

### 1. Whisper Detection — Logging, Cooldown, Highpass Filter (`whisper_phrase_detector.py`)
- Added detailed transcription logging in `_inference_loop`:
  - Every Whisper transcription result logged with text, confidence,
    no_speech_prob, peak, RMS, inference time, and rejection reason
  - "empty (no segments produced)" logged when Whisper returns nothing
- Added 3-second cooldown after detection to suppress duplicate triggers
- Added segmenter logging (VAD-gated vs peak-gated)
- Added 300Hz highpass filter (scipy butterworth 4th order) in audio callback
  to remove low-frequency motor rumble — helps marginally
- Changed VAD gating: always enforce hardware VAD when available (motor noise
  bypassed the old peak>0.10 threshold). Discovered firmware VAD is dead with
  motors running, so disabled VAD in demo.

### 2. DOA Circular Median & Outlier Rejection (`respeaker_doa_provider.py`)
- **Fixed critical bug**: `get_latched_direction()` used linear `np.median()` which
  gives wrong results near ±180° boundary
- Added `_circular_median()`: atan2-based shift-median-unshift
- Added `_reject_outliers_iqr()`: IQR on circularly-shifted angles
- Added diagnostic logging with sample counts and outlier stats

### 3. Rotation Calibration & Demo Fixes (`hear_and_rotate_demo.py`)
- **Rotation calibrated**: cmd_z=2.0 at 90°/s — **confirmed accurate on hardware**
  (90° test rotation matched expected angle)
- Added 8° deadzone to skip jitter rotations
- **Fixed critical bug**: `device=None` selected PulseAudio "default" (all zeros).
  Now auto-detects ReSpeaker by name via sounddevice.
- Disabled hardware VAD (permanently False with motors running)
- Set hop_duration_ms=1000 (was 200 — flooded inference with motor noise)
- Set confidence_threshold=0.30 (motor noise degrades logprobs)
- Set no_speech_threshold=0.75 (motor noise inflates no_speech_prob)
- Set mic_gain=40.0 (tested 15, 25, 40 — range limited by motor noise regardless)
- Added USB reset helper for ALSA "Broken pipe" recovery

### 4. Key Discoveries from Hardware Testing
- **ReSpeaker firmware VAD is dead when GO2 motors are running** — VOICEACTIVITY
  register permanently reads 0, DOAANGLE stuck at fixed value
- **Motor noise is structure-borne** (mechanical vibration through chassis), not
  airborne — broadband, peaks 0.65-0.92 after gain, unaffected by highpass filter
- **Detection range limited to ~1m** by motor noise SNR. Voice at 1m: peak=1.0,
  RMS=0.27-0.50. Voice at 2-3m: indistinguishable from noise.
- **Whisper correctly identifies "come here"** even in noise — confirmed multiple
  detections at conf 0.37-0.63
- **Rotation accuracy is good**: 90°/s at cmd_z=2.0 confirmed by manual test
- **"I am coming" plays on robot speaker** and Whisper hears it (conf=0.48-0.52)

## Files Changed

| File | Path (relative to come-here/) | Changes |
|------|------|---------|
| whisper_phrase_detector.py | come_here_audio/come_here_audio/ | +logging, +cooldown, +highpass filter, +empty segment detection |
| respeaker_doa_provider.py | come_here_audio/come_here_audio/ | +circular_median, +IQR outlier rejection, +logging |
| hear_and_rotate_demo.py | come_here_audio/scripts/ | rotation cal, ReSpeaker auto-detect, no VAD, thresholds |

## Calibration Changes (Final State)

| Parameter | Old Value | Final Value | File | Rationale |
|-----------|-----------|-------------|------|-----------|
| mic_gain | 25.0 | 40.0 | hear_and_rotate_demo.py | Maximize range (clipping OK, Whisper handles it) |
| confidence_threshold | 0.4 | 0.30 | hear_and_rotate_demo.py | Motor noise degrades logprobs to 0.28-0.63 |
| no_speech_threshold | 0.5 | 0.75 | hear_and_rotate_demo.py | Motor noise inflates no_speech_prob |
| cmd_z | 3.0 | 2.0 | hear_and_rotate_demo.py | Smoother, confirmed 90°/s on hardware |
| deg_per_sec | 200.0 | 90.0 | hear_and_rotate_demo.py | Confirmed accurate via manual 90° test |
| hop_duration_ms | 200 | 1000 | hear_and_rotate_demo.py | Prevents flooding inference with motor noise |
| mic_device | None | auto-detect | hear_and_rotate_demo.py | None selects PulseAudio (silence); auto-detect finds ReSpeaker |
| vad_check_fn | vad_recently_active | None | hear_and_rotate_demo.py | Firmware VAD dead with motors |
| min rotation dur | 0.2s | 0.3s | hear_and_rotate_demo.py | More reliable minimum turn |
| rotation deadzone | none | 8° | hear_and_rotate_demo.py | Skip jitter rotations |
| DOA median | np.median | circular_median | respeaker_doa_provider.py | Fix wraparound at ±180° |
| DOA outlier filter | none | IQR (1.5) | respeaker_doa_provider.py | Remove spurious firmware readings |
| detection cooldown | none | 3.0s | whisper_phrase_detector.py | Suppress duplicate triggers |
| highpass filter | none | 300Hz butter4 | whisper_phrase_detector.py | Remove low-freq motor rumble |

## Verification Performed

1. **Live hardware tests** (multiple runs, 60-120s each):
   - Detection confirmed: "come here" detected at conf=0.37, 0.51, 0.54, 0.57, 0.63
   - Rotation confirmed: 90° manual test at cmd_z=2.0 matched expected angle
   - Rotations executed: -109°, -113°, +9°, +171°, +172° — all completed correctly
   - "I am coming" voice response played on robot speaker
   - Latency: ~1.7s speech-to-rotate (1.2s inference + 0.5s pipeline)

2. **Calibration iterations on hardware**:
   - mic_gain: tested 15 (too low, silence), 25 (works close), 40 (no range improvement)
   - confidence: tested 0.45 (too tight, rejects valid), 0.30 (works)
   - no_speech: tested 0.50 (rejects real speech), 0.75 (works)
   - VAD: tested enabled (never fires with motors), disabled (works)
   - hop: tested 200ms (floods inference), 1000ms (works)

3. **Checksum verification** (final state):
   - hear_and_rotate_demo.py: `f2624dec5c8b525fd78e71ada707a7e5`
   - respeaker_doa_provider.py: `0bab4a1dde91884b4ebd8cb097991a2c`
   - whisper_phrase_detector.py: `94e15e9bd53c387922ec563df1c09e8f`

## Achieved Today

- **Detection works**: "come here" detected reliably at ~1m with motors running
- **Rotation calibrated**: 90°/s at cmd_z=2.0 confirmed on hardware
- Fixed circular median bug in DOA (real correctness issue)
- Fixed ReSpeaker device auto-detection (was selecting wrong device)
- Discovered firmware VAD is unusable with motors — documented and worked around
- Comprehensive diagnostic logging across full pipeline
- Multiple live test iterations with real calibration adjustments
- T7 ↔ Jetson synced and verified

## Still To Do

1. **Extend detection range** (currently ~1m with motors):
   - Physical: vibration-isolate ReSpeaker (rubber/foam standoffs)
   - Or: have dog lie down first (motors off → listen → stand → rotate)
   - Or: external mic on cable away from robot body
   - Software: spectral subtraction (estimate motor noise profile, subtract)

2. **CUDA inference**: CTranslate2 on CPU takes ~1.2s per inference. CUDA would
   cut to ~100ms. See TODO_ctranslate2_cuda.md.

3. **Custom DOA (GCC-PHAT)**: Firmware DOA dead with motors. Need custom DOA
   using raw 4-mic channels. See TODO_custom_doa.md.

4. **Vision integration**: YOLO person detection coded but untested on hardware.

5. **Behavior node rotation bridge**: ROS2 behavior_node needs Sport API bridge.

6. **False positive testing**: Need longer session to check if non-"come here"
   speech triggers false matches at conf>=0.30.

## Risks / Unknowns

- **~1m range is a hard limit with current hardware mounting** — motor vibration
  is structure-borne, no software filter can fix it. Physical isolation required
  for >2m range.
- **Firmware DOA unreliable with motors** — always returns same angle. Custom DOA
  (GCC-PHAT on raw mics) needed for accurate direction with motors running.
- **conf>=0.30 may cause false positives** — not tested with extended ambient
  speech. Monitor in future sessions.
- **USB ALSA pipe breaks** after demo crashes — need USB reset before each run.
  Software reset via pyusb works (no physical replug needed).

## Next Recommended Actions

1. **Vibration isolation**: Mount ReSpeaker on foam/rubber standoffs to extend range
2. **Try lie-down-then-listen**: StandDown (1005) → listen → RecoveryStand (1006) → rotate
3. **Custom DOA**: Implement GCC-PHAT using raw mic channels (firmware DOA is dead)
4. **CUDA build**: Build CTranslate2 with CUDA for ~10x faster inference
5. **Vision**: Test YOLO person detection on hardware

## How to Run

```bash
# USB reset first (if ALSA broken)
sshpass -p '123' ssh unitree@192.168.123.18 "echo '123' | sudo -S python3 -c \"
import fcntl,os,subprocess
r=subprocess.run(['lsusb'],capture_output=True,text=True)
for l in r.stdout.splitlines():
    if '2886:0018' in l:
        p=l.split(); fd=os.open(f'/dev/bus/usb/{p[1]}/{p[3].rstrip(chr(58))}',os.O_WRONLY)
        fcntl.ioctl(fd,21780,0); os.close(fd); print('USB reset OK')
\""

# Wait 3 seconds, then run demo
sleep 3
sshpass -p '123' ssh -o ServerAliveInterval=10 unitree@192.168.123.18 \
  "cd ~/come-here && source /opt/ros/humble/setup.bash && \
   source ~/go2_ws/install/setup.bash && \
   export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   export CYCLONEDDS_URI=file:///home/unitree/unitree_ros2/cyclonedds_ws/src/cyclonedds.xml && \
   PYTHONPATH=~/come-here/come_here_audio:\$PYTHONPATH \
   python3 -u come_here_audio/scripts/hear_and_rotate_demo.py"
```

## Networking
- **Jetson WiFi**: Connected to Carebear (192.168.8.240)
- **SSH used this session**: ethernet (192.168.123.18) — more reliable
- **iPhone hotspot**: no longer active
