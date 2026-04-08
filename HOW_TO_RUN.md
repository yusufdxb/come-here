# How to Run the Come-Here Demo

## Prerequisites
- Jetson Orin NX powered on and connected to hotspot WiFi
- GO2 robot powered on and standing (BalanceStand)
- ReSpeaker Mic Array plugged into Jetson USB
- PC connected to same hotspot (172.20.10.x subnet)

## SSH to Jetson
```bash
sshpass -p '<REDACTED>' ssh unitree@172.20.10.6
```

## Run the Demo
```bash
cd ~/come-here
source /opt/ros/humble/setup.bash
source ~/go2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/unitree/unitree_ros2/cyclonedds_ws/src/cyclonedds.xml
PYTHONPATH=~/come-here/come_here_audio:$PYTHONPATH python3 -u come_here_audio/scripts/hear_and_rotate_demo.py
```

## One-liner from PC
```bash
sshpass -p '<REDACTED>' ssh -o ServerAliveInterval=10 unitree@172.20.10.6 "cd ~/come-here && source /opt/ros/humble/setup.bash && source ~/go2_ws/install/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && export CYCLONEDDS_URI=file:///home/unitree/unitree_ros2/cyclonedds_ws/src/cyclonedds.xml && PYTHONPATH=~/come-here/come_here_audio:\$PYTHONPATH python3 -u come_here_audio/scripts/hear_and_rotate_demo.py"
```

## If ReSpeaker audio fails (Broken pipe error)
Unplug and replug the ReSpeaker USB cable, then retry.

## Syncing code from T7 to Jetson
```bash
sshpass -p '<REDACTED>' rsync -avz \
  --exclude 'build/' --exclude 'install/' --exclude 'log/' --exclude '.git/' \
  --exclude '*.whl' --exclude '*.wav' --exclude '*.jpg' --exclude 'models/' --exclude 'deps/' --exclude 'training/' --exclude '.planning/' \
  "/media/careslab/T7 Storage/come-here/come_here_audio/" \
  unitree@172.20.10.6:~/come-here/come_here_audio/
```

## ReSpeaker Tuning (resets on USB replug)
```bash
cd ~/usb_4_mic_array
python3 tuning.py AGCMAXGAIN 1000
python3 tuning.py AGCONOFF 1
```
