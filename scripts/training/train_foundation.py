"""
Foundation Training for IMI Medical LLM

Medical foundation fine-tuning of Mixtral 8x7B on combined medical datasets.
This creates a medically-specialized base model before DPO alignment and adapter training.

Training pipeline:
  Foundation Training (this script) ← you are here
  → DPO Safety Alignment (train_dpo.py)
  → Per-User Adapter Training (train_lora.py)

Key design decisions:
- QLoRA 4-bit NF4 with bfloat16 compute — fits on 1x A100 80GB
- Higher LoRA rank (r=64) than adapters — foundation needs broad capacity
- Target ALL Mixtral modules including MoE layers (w1, w2, w3)
- Sequence packing enabled for efficiency
- 2 epochs on combined medical corpus (~3M examples)
"""
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, load_dataset, concatenate_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================================
# FOUNDATION TRAINING CONFIG — Mixtral 8x7B specific
# ============================================================================

FOUNDATION_CONFIG = {
    # Model
    "base_model": "mistralai/Mixtral-8x7B-Instruct-v0.1",

    # LoRA — broader than adapters, includes MoE expert layers
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "w1", "w2", "w3",                          # MoE expert layers
    ],

    # Training
    "num_epochs": 2,
    "batch_size": 2,
    "gradient_accumulation_steps": 16,  # effective batch = 32
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "max_grad_norm": 0.3,
    "max_seq_length": 4096,
    "packing": True,  # Pack short sequences for efficiency

    # Precision
    "bf16": True,
    "fp16": False,

    # Optimizer
    "optim": "paged_adamw_32bit",

    # Saving
    "save_steps": 500,
    "save_total_limit": 3,
    "logging_steps": 50,

    # Monitoring
    "report_to": "wandb",
    "run_name": "imi-foundation-v1",
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

    # Load from train/ directory (merged by adapter type)
    if train_dir.exists():
        for f in sorted(train_dir.glob("*_train.json")):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                all_examples.extend(data)
                logger.info(f"  Loaded {len(data):,} examples from {f.name}")
            except Exception as e:
                logger.warning(f"  Failed to load {f.name}: {e}")

    # Also load from processed/ subdirectories
    if processed_dir.exists():
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
        logger.error("No training data found! Run data collection first:")
        logger.error("  python scripts/data_collection/collect_datasets.py")
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

    # Format for SFTTrainer — single 'text' column with Mixtral chat template
    formatted = []
    for ex in unique_examples:
        instruction = ex.get("instruction", "")
        input_text = ex.get("input", "")
        output = ex.get("output", "")

        if input_text:
            user_msg = f"{instruction}\n\n{input_text}"
        else:
            user_msg = instruction

        # Mixtral chat template
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
    """Run foundation training on combined medical corpus"""
    config = dict(FOUNDATION_CONFIG)
    if config_overrides:
        config.update(config_overrides)

    base_model = base_model or config["base_model"]
    data_path = Path(data_dir) if data_dir else DATA_DIR
    output_path = output_dir or str(MODELS_DIR / "foundation")

    logger.info("=" * 60)
    logger.info("IMI Foundation Training — Mixtral 8x7B Medical")
    logger.info("=" * 60)
    logger.info(f"Base model:    {base_model}")
    logger.info(f"Data dir:      {data_path}")
    logger.info(f"Output:        {output_path}")
    logger.info(f"LoRA rank:     {config['lora_r']} (alpha={config['lora_alpha']})")
    logger.info(f"Target modules: {config['target_modules']}")
    logger.info(f"Epochs:        {config['num_epochs']}")
    logger.info(f"Batch size:    {config['batch_size']} × {config['gradient_accumulation_steps']} = {config['batch_size'] * config['gradient_accumulation_steps']}")
    logger.info(f"Seq length:    {config['max_seq_length']}")
    logger.info(f"Packing:       {config['packing']}")

    # Load dataset
    dataset = load_foundation_dataset(data_path, max_examples)
    logger.info(f"Training on {len(dataset):,} examples")

    # QLoRA quantization
    logger.info("Loading model with QLoRA (4-bit NF4, bfloat16)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config — broader for foundation (includes MoE layers)
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Print trainable parameters
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Training arguments
    training_args = SFTConfig(
        output_dir=output_path,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        max_grad_norm=config["max_grad_norm"],
        optim=config["optim"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        logging_steps=config["logging_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        report_to=config["report_to"],
        run_name=config["run_name"],
        dataloader_num_workers=4,
        group_by_length=not config["packing"],  # Incompatible with packing
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        max_seq_length=config["max_seq_length"],
        packing=config["packing"],
        dataset_text_field="text",
    )

    # Train
    logger.info("Starting foundation training...")
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Save
    logger.info(f"Saving foundation model to {output_path}")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    # Save metadata
    metadata = {
        "training_type": "foundation",
        "base_model": base_model,
        "num_examples": len(dataset),
        "config": config,
    }
    with open(Path(output_path) / "foundation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 60)
    logger.info("Foundation training complete!")
    logger.info(f"Next step: python scripts/training/train_dpo.py train --foundation-path {output_path}")
    logger.info("=" * 60)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Foundation Training for IMI Medical LLM")
    parser.add_argument("--base-model", default=None, help="Base model path (default: Mixtral 8x7B)")
    parser.add_argument("--data-dir", default=None, help="Data directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--max-examples", type=int, default=None, help="Cap training examples")
    parser.add_argument("--resume-from", default=None, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lora-r", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--no-packing", action="store_true", help="Disable sequence packing")

    args = parser.parse_args()

    overrides = {}
    if args.epochs:
        overrides["num_epochs"] = args.epochs
    if args.lr:
        overrides["learning_rate"] = args.lr
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.lora_r:
        overrides["lora_r"] = args.lora_r
        overrides["lora_alpha"] = args.lora_r * 2
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
