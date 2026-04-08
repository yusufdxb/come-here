#!/usr/bin/env python3
"""Fine-tune Whisper with LoRA for "come here" wake phrase detection.

Uses HuggingFace transformers + PEFT (LoRA) to fine-tune Whisper on
recorded audio samples. Trains the model to accurately transcribe
"come here" in various conditions (noise, distance, accents).

The fine-tuned LoRA adapter is small (~few MB) and loads on top of
the base Whisper model at inference time.

Usage:
    # Fine-tune on recorded samples
    python3 finetune_whisper.py --data-dir data/ --output-dir output/

    # Resume training
    python3 finetune_whisper.py --data-dir data/ --output-dir output/ --resume

    # Use a larger base model
    python3 finetune_whisper.py --model openai/whisper-small.en --data-dir data/

Hardware: RTX 5070 (12GB VRAM) -- base.en with LoRA fits comfortably.
"""

import argparse
import os
import json

import numpy as np
import torch
from datasets import Audio, Dataset
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import soundfile as sf


def load_audio_dataset(data_dir: str) -> Dataset:
    """Load recorded WAV samples into a HuggingFace Dataset.

    Expected structure:
        data_dir/
            positive/   -- WAVs of "come here"
            negative/   -- WAVs of other speech/noise
    """
    samples = []
    pos_dir = os.path.join(data_dir, "positive")
    neg_dir = os.path.join(data_dir, "negative")

    if os.path.isdir(pos_dir):
        for fname in sorted(os.listdir(pos_dir)):
            if fname.endswith(".wav"):
                samples.append({
                    "audio": os.path.join(pos_dir, fname),
                    "transcription": "come here",
                })

    if os.path.isdir(neg_dir):
        for fname in sorted(os.listdir(neg_dir)):
            if fname.endswith(".wav"):
                # For negative samples, we want the model to transcribe
                # whatever was actually said (not "come here").
                # Using empty string as a placeholder -- ideally these
                # would have real transcriptions, but for wake-phrase
                # fine-tuning, teaching the model that these are NOT
                # "come here" is the key goal.
                samples.append({
                    "audio": os.path.join(neg_dir, fname),
                    "transcription": "",
                })

    if not samples:
        raise ValueError(
            f"No WAV files found in {data_dir}/positive/ or {data_dir}/negative/. "
            "Run record_samples.py first."
        )

    print(f"Loaded {len(samples)} samples "
          f"({sum(1 for s in samples if s['transcription'])} positive, "
          f"{sum(1 for s in samples if not s['transcription'])} negative)")

    ds = Dataset.from_list(samples)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


def prepare_dataset(batch, processor):
    """Preprocess a batch for Whisper training."""
    audio = batch["audio"]
    input_features = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="np",
    ).input_features[0]

    labels = processor.tokenizer(batch["transcription"]).input_ids

    batch["input_features"] = input_features
    batch["labels"] = labels
    return batch


class DataCollator:
    """Collate processed samples into training batches."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present (Whisper adds it during generation)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA")
    # Resolve default model path: use local cache on T7 if available
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_model = os.path.join(script_dir, "..", "models", "whisper-base.en")
    default_model = local_model if os.path.isdir(local_model) else "openai/whisper-base.en"

    parser.add_argument("--model", default=default_model,
                        help="Base Whisper model path or HF name")
    parser.add_argument("--data-dir", default="data", help="Training data directory")
    parser.add_argument("--output-dir", default="output", help="Output directory for adapter")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load processor and model
    print(f"\nLoading {args.model}...")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Apply LoRA
    print(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and preprocess data
    print(f"\nLoading data from {args.data_dir}/...")
    dataset = load_audio_dataset(args.data_dir)
    dataset = dataset.map(
        lambda b: prepare_dataset(b, processor),
        remove_columns=dataset.column_names,
    )

    # Split into train/eval (90/10)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Train: {len(split['train'])} samples, Eval: {len(split['test'])} samples")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=10,
        predict_with_generate=True,
        generation_max_length=30,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollator(processor),
        processing_class=processor.feature_extractor,
    )

    # Train
    print("\nStarting training...")
    if args.resume and os.path.isdir(args.output_dir):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save the LoRA adapter
    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    # Save training config for reproducibility
    config = {
        "base_model": args.model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": device,
    }
    with open(os.path.join(adapter_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nAdapter saved to {adapter_dir}/")
    print(f"Size: {sum(os.path.getsize(os.path.join(adapter_dir, f)) for f in os.listdir(adapter_dir)) / 1e6:.1f} MB")
    print("\nTo use in come-here system:")
    print(f"  Set whisper_adapter_path: {os.path.abspath(adapter_dir)}")
    print("  Set wake_detector: whisper")


if __name__ == "__main__":
    main()
