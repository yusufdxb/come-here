#!/bin/bash
# Setup script for come-here project
# Run this on any machine with the T7 plugged in.
#
# Usage:
#   cd /media/yusuf/T7\ Storage/come-here
#   bash setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "=== come-here setup ==="
echo "Project dir: $SCRIPT_DIR"

# 1. Install Python deps (from local cache if available)
echo ""
echo "--- Installing Python dependencies ---"
if [ -d "$SCRIPT_DIR/deps" ] && [ "$(ls -A "$SCRIPT_DIR/deps" 2>/dev/null)" ]; then
    echo "Installing from local cache (deps/)..."
    pip install --no-index --find-links "$SCRIPT_DIR/deps" \
        faster-whisper sounddevice numpy transformers peft accelerate datasets torch torchaudio 2>&1 || {
        echo "Local install failed, falling back to network..."
        pip install faster-whisper sounddevice numpy transformers peft accelerate datasets
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    }
else
    echo "No local cache, installing from network..."
    pip install faster-whisper sounddevice numpy transformers peft accelerate datasets
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
fi

# 2. Check GPU
echo ""
echo "--- GPU check ---"
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')
else:
    print('WARNING: No CUDA GPU detected. Whisper will run on CPU (slower).')
"

# 3. Check audio devices
echo ""
echo "--- Audio devices ---"
python3 -c "
import sounddevice as sd
print('Available input devices:')
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        print(f'  [{i}] {dev[\"name\"]} ({dev[\"max_input_channels\"]}ch, {int(dev[\"default_samplerate\"])}Hz)')
print(f'Default input: [{sd.default.device[0]}] {sd.query_devices(sd.default.device[0])[\"name\"]}')
" 2>/dev/null || echo "WARNING: sounddevice not working. Check audio drivers."

# 4. Check ROS 2
echo ""
echo "--- ROS 2 check ---"
if [ -f /opt/ros/humble/setup.bash ]; then
    echo "ROS 2 Humble found."
    echo "To build:"
    echo "  source /opt/ros/humble/setup.bash"
    echo "  cd $SCRIPT_DIR"
    echo "  colcon build --symlink-install"
    echo "  source install/setup.bash"
else
    echo "WARNING: ROS 2 Humble not found at /opt/ros/humble/"
    echo "Whisper and training scripts work standalone without ROS."
fi

# 5. Verify Whisper model cache
echo ""
echo "--- Model cache ---"
if [ -d "$SCRIPT_DIR/models/whisper-base.en" ]; then
    echo "HF whisper-base.en: cached ($(du -sh "$SCRIPT_DIR/models/whisper-base.en" | cut -f1))"
else
    echo "WARNING: HF model not cached. Will download on first use."
fi
if [ -d "$SCRIPT_DIR/models/faster-whisper-base.en" ]; then
    echo "faster-whisper base.en: cached ($(du -sh "$SCRIPT_DIR/models/faster-whisper-base.en" | cut -f1))"
else
    echo "WARNING: faster-whisper model not cached. Will download on first use."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Quick start:"
echo "  # Record samples"
echo "  cd $SCRIPT_DIR/training"
echo "  python3 record_samples.py --output-dir data/ --num-positive 30 --num-negative 30"
echo ""
echo "  # Fine-tune"
echo "  python3 finetune_whisper.py --data-dir data/ --output-dir output/"
echo ""
echo "  # Live test"
echo "  python3 evaluate.py --adapter-dir output/lora_adapter/ --live"
echo ""
echo "  # ROS 2 launch (after building)"
echo "  ros2 launch come_here_bringup come_here.launch.py"
