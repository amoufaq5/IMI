"""
Mistral 7B QLoRA Fine-tuning on Medical Corpus
===============================================

Fine-tunes mistralai/Mistral-7B-Instruct-v0.3 using QLoRA (4-bit NF4
quantization + LoRA adapters) on two data formats:

  general_knowledge — Raw medical text passages (causal LM style)
  instruction       — Medical Q&A / instruction-following pairs

QLoRA keeps the base model in 4-bit while training only the LoRA adapter
weights in BF16. Mistral 7B in 4-bit fits in ~6 GB VRAM; full QLoRA run
with batch=4, seq=2048 fits comfortably in a single A100 40GB.

GPU tier presets:
  --gpu-tier RTX3090     batch=2, seq=1024, grad_accum=8   (24 GB VRAM)
  --gpu-tier A100_40GB   batch=4, seq=2048, grad_accum=4   (40 GB VRAM) ← recommended
  --gpu-tier A100_80GB   batch=8, seq=4096, grad_accum=2   (80 GB VRAM)
  --gpu-tier H100_80GB   batch=16, seq=4096, grad_accum=1  (80 GB VRAM, fast)

Usage:
    # Demo: 100 examples, 10 steps — verify setup works before full training
    python scripts/training/finetune_mixtral.py --demo

    # Full training with recommended GPU
    python scripts/training/finetune_mixtral.py --gpu-tier A100_40GB

    # Custom settings
    python scripts/training/finetune_mixtral.py \\
        --gpu-tier A100_80GB \\
        --epochs 3 \\
        --output-dir models/mistral-medical-v1 \\
        --data-format both        # "general_knowledge", "instruction", or "both"

    # Resume from checkpoint
    python scripts/training/finetune_mixtral.py \\
        --gpu-tier A100_80GB \\
        --resume-from models/mistral-medical-v1/checkpoint-500

Prerequisites (install with scripts/install_training.sh):
    torch==2.2.0  transformers==4.42.0  peft==0.11.1  trl==0.9.6
    bitsandbytes==0.43.1  datasets==2.19.0  accelerate==0.30.0
"""
import os
import sys
import json
import logging
import argparse
import random
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# =============================================================================
# GPU TIER CONFIGS
# Mistral 7B is much smaller than Mixtral 8x7B — fits in a single GPU.
# effective_batch = per_device_batch × grad_accum_steps  (× num_gpus if multi-GPU)
# =============================================================================

GPU_TIERS: Dict[str, Dict[str, Any]] = {
    "RTX3090": {
        # 24 GB VRAM — Mistral 7B 4-bit (~6 GB) + activations fits comfortably
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,     # effective batch = 16
        "max_seq_length": 1024,
        "gradient_checkpointing": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "note": "RTX 3090 24 GB — budget option, works for Mistral 7B.",
    },
    "A100_40GB": {
        # 40 GB VRAM — recommended for Mistral 7B QLoRA
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,     # effective batch = 16
        "max_seq_length": 2048,
        "gradient_checkpointing": True,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "note": "A100 40 GB — recommended. Best price/performance for Mistral 7B.",
    },
    "A100_80GB": {
        # 80 GB VRAM — comfortable; use larger seq/batch for throughput
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,     # effective batch = 16
        "max_seq_length": 4096,
        "gradient_checkpointing": True,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "note": "A100 80 GB — plenty of headroom for Mistral 7B.",
    },
    "H100_80GB": {
        # 80 GB VRAM + faster BF16 tensor cores
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 1,     # effective batch = 16
        "max_seq_length": 4096,
        "gradient_checkpointing": False,       # enough VRAM to skip checkpointing
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "note": "H100 80 GB — fastest option for Mistral 7B.",
    },
    "DEMO": {
        # Minimal config: verify pipeline works in <5 minutes
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 512,
        "gradient_checkpointing": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "note": "Demo mode — 100 examples, 10 steps.",
    },
}

# LoRA target modules for Mistral 7B (dense transformer — no MoE)
# Attention projections + SwiGLU FFN projections
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",        # SwiGLU FFN (replaces Mixtral w1/w2/w3)
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_format(data_dir: Path, fmt: str) -> List[Dict[str, Any]]:
    """Load a single data format (general_knowledge or instruction)."""
    final_dir = data_dir / "final"
    examples = []

    for split in ("train",):
        path = final_dir / f"{fmt}_{split}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            examples.extend(data)
            logger.info(f"  Loaded {len(data):,} examples from {path.name}")
        else:
            logger.warning(f"  Not found: {path}")
            logger.warning(f"  Run prepare_medical_data.py first.")

    return examples


def format_general_knowledge(example: Dict[str, Any]) -> str:
    """
    Format a general_knowledge example for causal LM training.
    No chat template — pure text continuation for factual grounding.
    """
    text = example.get("text", "").strip()
    if not text:
        return ""
    return f"<s>{text}</s>"


def format_instruction(example: Dict[str, Any]) -> str:
    """
    Format an instruction example using the Mixtral [INST] chat template.
    Input is optional — if present it is appended to the instruction.
    """
    instruction = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    output = example.get("output", "").strip()

    if not instruction or not output:
        return ""

    user_msg = f"{instruction}\n\n{inp}".strip() if inp else instruction
    return f"<s>[INST] {user_msg} [/INST] {output}</s>"


def build_hf_dataset(examples: List[Dict[str, Any]], fmt: str, max_examples: Optional[int] = None):
    """Convert list of dicts into a HuggingFace Dataset with a 'text' column."""
    from datasets import Dataset

    formatted = []
    for ex in examples:
        if fmt == "general_knowledge":
            text = format_general_knowledge(ex)
        else:
            text = format_instruction(ex)

        if text:
            formatted.append({"text": text})

    if max_examples:
        formatted = formatted[:max_examples]

    logger.info(f"  {fmt}: {len(formatted):,} formatted examples")
    return Dataset.from_list(formatted)


def load_training_dataset(
    data_dir: Path,
    data_format: str,          # "general_knowledge", "instruction", or "both"
    max_examples: Optional[int] = None,
):
    """Load and combine data according to requested format(s)."""
    from datasets import concatenate_datasets

    if data_format == "both":
        formats = ["general_knowledge", "instruction"]
    else:
        formats = [data_format]

    datasets = []
    for fmt in formats:
        examples = load_data_format(data_dir, fmt)
        if examples:
            random.shuffle(examples)
            ds = build_hf_dataset(examples, fmt, max_examples)
            datasets.append(ds)
        else:
            logger.warning(f"No data found for format '{fmt}' — skipping.")

    if not datasets:
        raise ValueError(
            "No training data found!\n"
            "  Run: python scripts/training/prepare_medical_data.py\n"
            "  Then: python scripts/data_collection/collect_datasets.py"
        )

    if len(datasets) == 1:
        combined = datasets[0]
    else:
        combined = concatenate_datasets(datasets)

    # Shuffle combined dataset
    combined = combined.shuffle(seed=42)

    logger.info(f"Total training examples: {len(combined):,}")
    return combined


# =============================================================================
# MODEL & QUANTIZATION
# =============================================================================

def load_model_and_tokenizer(base_model: str, tier_cfg: Dict[str, Any]):
    """
    Load Mistral 7B in 4-bit NF4 QLoRA mode.
    The base weights stay frozen in 4-bit; only LoRA adapters are trained in BF16.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

    logger.info(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # required for SFT with packing disabled
    tokenizer.model_max_length = tier_cfg["max_seq_length"]

    # 4-bit NF4 quantization config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # Normal Float 4 — best for QLoRA
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,     # Nested quantization → saves ~0.4 GB
    )

    logger.info("Loading model in 4-bit NF4 (QLoRA)...")
    logger.info("  Base weights: frozen 4-bit  |  LoRA adapters: trainable BF16")

    # Check if flash-attn is available
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        logger.info("  Attention: Flash Attention 2 (installed)")
    except ImportError:
        attn_impl = "eager"
        logger.info("  Attention: standard eager (flash-attn not installed)")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )

    # Prepare for k-bit training (casts norms, sets requires_grad)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=tier_cfg["gradient_checkpointing"],
    )

    # Attach LoRA adapters
    lora_config = LoraConfig(
        r=tier_cfg["lora_r"],
        lora_alpha=tier_cfg["lora_alpha"],
        lora_dropout=tier_cfg["lora_dropout"],
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


# =============================================================================
# TRAINING
# =============================================================================

def build_training_args(
    output_dir: str,
    tier_cfg: Dict[str, Any],
    num_epochs: int,
    learning_rate: float,
    demo_mode: bool,
    report_to: str = "none",
):
    """Build HuggingFace TrainingArguments from tier config."""
    from transformers import TrainingArguments

    save_steps = 10 if demo_mode else 200
    max_steps = 10 if demo_mode else -1        # -1 = run full epochs

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=tier_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tier_cfg["gradient_accumulation_steps"],
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        optim="paged_adamw_32bit",             # QLoRA standard optimizer (saves VRAM)
        fp16=False,
        bf16=True,                             # BF16 for LoRA adapter training
        logging_steps=5 if demo_mode else 25,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        report_to=report_to,
        run_name="imi-mistral7b-medical-qlora",
        gradient_checkpointing=tier_cfg["gradient_checkpointing"],
        dataloader_num_workers=2,              # Keep low to avoid CPU memory issues
        dataloader_pin_memory=True,
        group_by_length=True,                  # Reduces padding waste
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )


def run_training(
    base_model: str,
    data_dir: Path,
    output_dir: str,
    tier: str,
    data_format: str,
    num_epochs: int,
    learning_rate: float,
    max_examples: Optional[int],
    resume_from: Optional[str],
    demo_mode: bool,
    report_to: str,
):
    from trl import SFTTrainer, SFTConfig

    tier_cfg = GPU_TIERS[tier]

    logger.info("=" * 60)
    logger.info("  IMI Mistral 7B — QLoRA Medical Fine-tuning")
    logger.info("=" * 60)
    logger.info(f"  Base model:      {base_model}")
    logger.info(f"  GPU tier:        {tier} — {tier_cfg['note']}")
    logger.info(f"  Data format:     {data_format}")
    logger.info(f"  Seq length:      {tier_cfg['max_seq_length']}")
    logger.info(f"  Batch (device):  {tier_cfg['per_device_train_batch_size']}")
    logger.info(f"  Grad accum:      {tier_cfg['gradient_accumulation_steps']}")
    logger.info(f"  Effective batch: "
                f"{tier_cfg['per_device_train_batch_size'] * tier_cfg['gradient_accumulation_steps']}")
    logger.info(f"  LoRA r/alpha:    {tier_cfg['lora_r']}/{tier_cfg['lora_alpha']}")
    logger.info(f"  Epochs:          {num_epochs}")
    logger.info(f"  Learning rate:   {learning_rate}")
    logger.info(f"  Output:          {output_dir}")
    if demo_mode:
        logger.info("  MODE:            DEMO (10 steps, 100 examples)")
    logger.info("=" * 60)

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"  GPU: {gpu_count}× {gpu_name} ({gpu_mem:.0f} GB)")

        if gpu_mem < 16:
            logger.warning(
                f"  VRAM ({gpu_mem:.0f} GB) may be too low for Mistral 7B QLoRA. "
                "Minimum recommended: 16 GB. Expect OOM if seq_len > 512."
            )
    else:
        logger.error("  No CUDA GPU detected! QLoRA training requires GPU.")
        sys.exit(1)

    # Load data
    logger.info("\nLoading training data...")
    cap = 100 if demo_mode else max_examples
    dataset = load_training_dataset(data_dir, data_format, max_examples=cap)

    # Load model + tokenizer
    logger.info("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(base_model, tier_cfg)

    # Training arguments
    training_args = build_training_args(
        output_dir=output_dir,
        tier_cfg=tier_cfg,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        demo_mode=demo_mode,
        report_to=report_to,
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=tier_cfg["max_seq_length"],
        packing=False,   # packing=False avoids tokenizer edge cases in QLoRA
    )

    # Train
    logger.info("\nStarting training...")
    if resume_from:
        logger.info(f"  Resuming from: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Save
    logger.info(f"\nSaving LoRA adapter to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metadata
    meta = {
        "training_type": "qlora_4bit_nf4",
        "base_model": base_model,
        "gpu_tier": tier,
        "data_format": data_format,
        "lora_r": tier_cfg["lora_r"],
        "lora_alpha": tier_cfg["lora_alpha"],
        "lora_target_modules": LORA_TARGET_MODULES,
        "max_seq_length": tier_cfg["max_seq_length"],
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "num_examples": len(dataset),
        "demo_mode": demo_mode,
    }
    with open(Path(output_dir) / "training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("=" * 60)
    if demo_mode:
        logger.info("  Demo training complete!")
        logger.info("  The pipeline works. Run without --demo for full training.")
    else:
        logger.info("  Training complete!")
        logger.info(f"  Adapter saved to: {output_dir}")
        logger.info(f"  Load adapter with:")
        logger.info(f"    from peft import PeftModel")
        logger.info(f"    model = PeftModel.from_pretrained(base_model, '{output_dir}')")
    logger.info("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mistral 7B QLoRA Medical Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--demo", action="store_true",
        help="Demo mode: 100 examples, 10 steps. Verify setup before full training.",
    )
    parser.add_argument(
        "--gpu-tier", default="A100_40GB",
        choices=list(GPU_TIERS.keys()),
        help="GPU preset (default: A100_40GB). Use RTX3090 for budget training.",
    )
    parser.add_argument(
        "--data-format", default="both",
        choices=["general_knowledge", "instruction", "both"],
        help=(
            "Training data format (default: both). "
            "'general_knowledge' = raw text passages. "
            "'instruction' = Q&A / instruction-following pairs. "
            "'both' = combined."
        ),
    )
    parser.add_argument(
        "--base-model", default=BASE_MODEL,
        help=f"HuggingFace model ID (default: {BASE_MODEL})",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Path to data directory (default: project root /data)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to save the LoRA adapter (default: models/mistral-medical-qlora)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Cap training examples (default: use all available)",
    )
    parser.add_argument(
        "--resume-from", default=None,
        help="Path to a checkpoint directory to resume from",
    )
    parser.add_argument(
        "--report-to", default="none",
        choices=["none", "wandb", "tensorboard"],
        help="Experiment tracker (default: none). Use 'wandb' if wandb is installed.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tier = "DEMO" if args.demo else args.gpu_tier
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    output_dir = args.output_dir or str(
        MODELS_DIR / ("mistral-medical-demo" if args.demo else "mistral-medical-qlora")
    )

    run_training(
        base_model=args.base_model,
        data_dir=data_dir,
        output_dir=output_dir,
        tier=tier,
        data_format=args.data_format,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        max_examples=args.max_examples,
        resume_from=args.resume_from,
        demo_mode=args.demo,
        report_to=args.report_to,
    )


if __name__ == "__main__":
    main()
