"""
Foundation Training for IMI Medical LLM — Full Fine-Tuning on 2×H100

Full fine-tuning of Mistral 7B on combined medical datasets.
Configured for 2× NVIDIA H100 80GB GPUs.

Training pipeline:
  Foundation Training (this script) ← you are here
  → DPO Safety Alignment (train_dpo.py)

Key design decisions:
- Full fine-tuning — all parameters updated for maximum quality
- BFloat16 mixed precision — native H100 support
- 2×H100 multi-GPU via DDP / FSDP
- SDPA (PyTorch native Scaled Dot-Product Attention) for fast training
- Gradient checkpointing for memory efficiency
- Sequence packing enabled for throughput maximization
"""
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================================
# 2×H100 FULL FINE-TUNE CONFIG
# ============================================================================

H100_FULL_FT_CONFIG = {
    # Model
    "base_model": "mistralai/Mistral-7B-Instruct-v0.3",

    # Training
    "num_epochs": 2,
    "per_device_batch_size": 4,
    "gradient_accumulation_steps": 4,   # effective batch = 4 * 4 * 2 GPUs = 32
    "learning_rate": 2e-5,              # Lower LR for full fine-tuning
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "max_grad_norm": 1.0,
    "max_seq_length": 4096,
    "packing": True,
    "weight_decay": 0.01,

    # Precision
    "bf16": True,
    "fp16": False,
    "tf32": True,

    # Optimizer — fused AdamW for H100 (no paged/8bit needed for full FT)
    "optim": "adamw_torch_fused",

    # Memory optimization
    "gradient_checkpointing": True,
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,

    # Saving
    "save_steps": 200,
    "save_total_limit": 3,
    "logging_steps": 25,

    # Monitoring
    "report_to": "none",
    "run_name": "imi-foundation-2xh100-full-ft",
}


def load_foundation_dataset(data_dir: Path, max_examples: Optional[int] = None) -> Dataset:
    """
    Load and combine all processed datasets for foundation training.
    Merges data from all adapter types into a single training corpus.
    """
    logger.info("Loading foundation training data...")

    all_examples = []
    train_dir = data_dir / "train"
    processed_dir = data_dir / "processed"
    final_dir = data_dir / "final"

    # Prefer final/ (preprocessed/validated data) over raw processed/
    if final_dir.exists():
        for f in sorted(final_dir.glob("*_train.json")):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                all_examples.extend(data)
                logger.info(f"  Loaded {len(data):,} examples from final/{f.name}")
            except Exception as e:
                logger.warning(f"  Failed to load {f.name}: {e}")

    # Fallback: load from train/ directory
    if not all_examples and train_dir.exists():
        for f in sorted(train_dir.glob("*_train.json")):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                all_examples.extend(data)
                logger.info(f"  Loaded {len(data):,} examples from train/{f.name}")
            except Exception as e:
                logger.warning(f"  Failed to load {f.name}: {e}")

    # Also load from processed/ subdirectories if needed
    if not all_examples and processed_dir.exists():
        for adapter_dir in sorted(processed_dir.iterdir()):
            if adapter_dir.is_dir():
                for f in sorted(adapter_dir.glob("*.json")):
                    try:
                        with open(f) as fp:
                            data = json.load(fp)
                        all_examples.extend(data)
                        logger.info(f"  Loaded {len(data):,} examples from {adapter_dir.name}/{f.name}")
                    except Exception as e:
                        logger.warning(f"  Failed to load {f.name}: {e}")

    if not all_examples:
        logger.error("No training data found! Run data collection and preprocessing first:")
        logger.error("  python scripts/data_collection/collect_datasets.py")
        logger.error("  python scripts/training/prepare_data.py")
        raise ValueError("No training data available")

    # Deduplicate by instruction+output hash
    seen = set()
    unique_examples = []
    for ex in all_examples:
        key = hash((ex.get("instruction", ""), ex.get("output", "")))
        if key not in seen:
            seen.add(key)
            unique_examples.append(ex)

    logger.info(f"Total: {len(all_examples):,} → {len(unique_examples):,} unique examples")

    # Shuffle
    import random
    random.shuffle(unique_examples)

    # Cap if requested
    if max_examples and len(unique_examples) > max_examples:
        unique_examples = unique_examples[:max_examples]
        logger.info(f"Capped to {max_examples:,} examples")

    # Format for SFTTrainer — single 'text' column with Mistral chat template
    formatted = []
    for ex in unique_examples:
        instruction = ex.get("instruction", "")
        input_text = ex.get("input", "")
        output = ex.get("output", "")

        if input_text:
            user_msg = f"{instruction}\n\n{input_text}"
        else:
            user_msg = instruction

        # Mistral chat template
        text = f"<s>[INST] {user_msg} [/INST] {output}</s>"
        formatted.append({"text": text})

    return Dataset.from_list(formatted)


def train_foundation(
    base_model: str = None,
    data_dir: str = None,
    output_dir: str = None,
    max_examples: Optional[int] = None,
    resume_from: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
):
    """Run full fine-tuning on 2×H100"""
    config = dict(H100_FULL_FT_CONFIG)
    if config_overrides:
        config.update(config_overrides)

    base_model = base_model or config["base_model"]
    data_path = Path(data_dir) if data_dir else DATA_DIR
    output_path = output_dir or str(MODELS_DIR / "foundation")

    logger.info("=" * 60)
    logger.info("IMI Foundation Training — 2×H100 Full Fine-Tuning")
    logger.info("=" * 60)
    logger.info(f"Base model:         {base_model}")
    logger.info(f"Data dir:           {data_path}")
    logger.info(f"Output:             {output_path}")
    logger.info(f"Training mode:      Full fine-tuning (all parameters)")
    logger.info(f"Precision:          BFloat16 mixed precision")
    logger.info(f"Epochs:             {config['num_epochs']}")
    logger.info(f"Per-device batch:   {config['per_device_batch_size']}")
    logger.info(f"Grad accumulation:  {config['gradient_accumulation_steps']}")
    logger.info(f"Effective batch:    {config['per_device_batch_size'] * config['gradient_accumulation_steps']} × num_gpus")
    logger.info(f"Learning rate:      {config['learning_rate']}")
    logger.info(f"Seq length:         {config['max_seq_length']}")
    logger.info(f"Packing:            {config['packing']}")
    logger.info(f"Grad checkpointing: {config['gradient_checkpointing']}")
    logger.info(f"Optimizer:          {config['optim']}")

    # Detect GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU:                {gpu_count}x {gpu_name} ({gpu_mem:.0f}GB)")
    else:
        logger.warning("No CUDA GPU detected! Training requires GPU.")

    # Enable TF32 for H100 matmul acceleration
    if config.get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32:               Enabled (H100 acceleration)")

    # Load dataset
    dataset = load_foundation_dataset(data_path, max_examples)
    logger.info(f"Training on {len(dataset):,} examples")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = config["max_seq_length"]

    # Load model in BFloat16 — full precision, no quantization
    logger.info("Loading model in BFloat16 (full fine-tuning)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,} (all trainable)")

    # SFTConfig
    training_args = SFTConfig(
        output_dir=output_path,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        max_grad_norm=config["max_grad_norm"],
        weight_decay=config["weight_decay"],
        optim=config["optim"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        logging_steps=config["logging_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        report_to=config["report_to"],
        run_name=config["run_name"],
        dataloader_num_workers=config["dataloader_num_workers"],
        dataloader_pin_memory=config.get("dataloader_pin_memory", True),
        gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )

    # SFTTrainer — full fine-tuning, no PEFT
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        packing=config["packing"],
    )

    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    trainable_pct = 100 * trainable_params / total_params
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")

    # Train
    logger.info("Starting full fine-tuning on 2×H100...")
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Save full model
    logger.info(f"Saving model to {output_path}")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    # Save metadata
    metadata = {
        "training_type": "full_fine_tuning",
        "training_hardware": "2xH100",
        "base_model": base_model,
        "num_examples": len(dataset),
        "config": config,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_pct": trainable_pct,
        "precision": "bfloat16",
    }
    with open(Path(output_path) / "foundation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 60)
    logger.info("Foundation training complete! (Full fine-tuning on 2×H100)")
    logger.info(f"Next step: python scripts/training/train_dpo.py train --foundation-path {output_path}")
    logger.info("=" * 60)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Foundation Training — 2×H100 Full Fine-Tuning")
    parser.add_argument("--base-model", default=None, help="Base model path (default: Mistral 7B)")
    parser.add_argument("--data-dir", default=None, help="Data directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--max-examples", type=int, default=None, help="Cap training examples")
    parser.add_argument("--resume-from", default=None, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device batch size")
    parser.add_argument("--no-packing", action="store_true", help="Disable sequence packing")
    parser.add_argument("--deepspeed", default=None, help="Path to DeepSpeed config JSON")

    args = parser.parse_args()

    overrides = {}
    if args.epochs:
        overrides["num_epochs"] = args.epochs
    if args.lr:
        overrides["learning_rate"] = args.lr
    if args.batch_size:
        overrides["per_device_batch_size"] = args.batch_size
    if args.no_packing:
        overrides["packing"] = False

    train_foundation(
        base_model=args.base_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        resume_from=args.resume_from,
        config_overrides=overrides if overrides else None,
    )


if __name__ == "__main__":
    main()
