# Whisper Fine-Tuning for "Come Here" Detection

Fine-tunes Whisper (base.en) with LoRA on the RTX 5070 to improve
"come here" detection accuracy in noisy/real-world conditions.

## Quick Start

```bash
# 1. Record training samples (need a mic)
python3 record_samples.py --output-dir data/ --num-positive 30 --num-negative 30

# 2. Fine-tune (runs on GPU, ~0.15 GB VRAM for base.en)
python3 finetune_whisper.py --data-dir data/ --output-dir output/ --epochs 10

# 3. Evaluate
python3 evaluate.py --adapter-dir output/lora_adapter/ --data-dir data/

# 4. Compare base vs fine-tuned
python3 evaluate.py --adapter-dir output/lora_adapter/ --data-dir data/ --compare

# 5. Live mic test
python3 evaluate.py --adapter-dir output/lora_adapter/ --live
```

## Recording Tips

- Record positive samples at different distances, volumes, and speeds
- Include variations: "come here", "come here!", whispering, shouting
- For negative samples: other commands ("go there", "stop"), conversation,
  background noise, silence, clapping, footsteps
- More samples = better. 30+ of each is a good start, 100+ is better
- Record in the actual environment where the robot will operate (lab) for best results

## Using the Fine-Tuned Model

Set in `audio_params.yaml`:
```yaml
audio_node:
  ros__parameters:
    wake_detector: whisper
    whisper_device: cuda
    whisper_adapter_path: /path/to/output/lora_adapter
```

## Model Sizes

| Model | Params | VRAM (LoRA) | Speed | Accuracy |
|-------|--------|-------------|-------|----------|
| tiny.en | 39M | ~0.08 GB | fastest | lower |
| base.en | 74M | ~0.15 GB | fast | good |
| small.en | 244M | ~0.5 GB | moderate | better |
| medium.en | 769M | ~1.5 GB | slower | best |

All fit comfortably on the RTX 5070 (12 GB). Start with base.en.

## What This Does NOT Cover

- Direction-of-arrival estimation (needs mic array hardware)
- Real-time latency optimization for deployment on Jetson
- Noise-specific augmentation (worth adding once lab environment is known)
