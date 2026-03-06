"""
Foundation Training for IMI Medical LLM — Full Fine-Tuning on 8× A100 80GB

Full-parameter fine-tuning of Mixtral 8x7B on the combined medical corpus.
Designed for 8× A100 80GB with DeepSpeed ZeRO Stage 3.

Training pipeline:
  Foundation Training (this script) ← you are here
  → Adapter Training (finetune_mixtral.py)

Key design decisions:
- FULL fine-tuning (no LoRA/QLoRA) — all 46.7B parameters updated
- BFloat16 precision — A100/H100 native
- DeepSpeed ZeRO Stage 3 — shards params + gradients + optimizer across 8 GPUs
- Flash Attention 2 — with eager fallback if not installed
- Gradient checkpointing enabled — mandatory for A100 full FT
- Sequence packing — maximises throughput
- paged_adamw_32bit — memory-efficient optimizer compatible with ZeRO

Run command (8× A100 80GB):
    torchrun --nproc_per_node=8 scripts/training/train_foundation.py \\
        --deepspeed configs/deepspeed_zero3.json

Resume from checkpoint:
    torchrun --nproc_per_node=8 scripts/training/train_foundation.py \\
        --deepspeed configs/deepspeed_zero3.json \\
        --resume-from models/foundation/checkpoint-1000
"""
import hashlib
import json
import logging
import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


# =============================================================================
# A100 80GB CONFIG — 8-GPU DeepSpeed ZeRO Stage 3
#
# Memory budget per GPU (80 GB):
#   Model shard (ZeRO-3):   46.7B × 2B / 8 GPUs  ≈ 11.7 GB
#   Gradient shard:                                ≈ 11.7 GB
#   Optimizer shard (32-bit AdamW):                ≈ 23.4 GB
#   Activations (batch=4, seq=2048, grad_ckpt):   ≈ 15–20 GB
#   Buffers / NCCL overhead:                       ≈  5 GB
#   TOTAL:                                         ≈ 68–72 GB  ✓ safe margin
# =============================================================================

A100_8GPU_CONFIG = {
    "base_model": "mistralai/Mixtral-8x7B-Instruct-v0.1",

    # Training
    "num_epochs": 1,
    "batch_size": 4,                    # per-device; effective = 4 × 4 accum × 8 GPUs = 128
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,             # lower than QLoRA — full FT is more sensitive
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "max_grad_norm": 1.0,
    "max_seq_length": 2048,
    "packing": True,                   # sequence packing for throughput

    # Precision
    "bf16": True,
    "fp16": False,
    "tf32": True,                      # A100 and H100 Tensor Cores

    # Optimizer — must be compatible with DeepSpeed ZeRO
    # adamw_torch_fused conflicts with ZeRO; use adamw_torch instead
    "optim": "adamw_torch",

    # Memory — gradient checkpointing is mandatory for A100 full FT
    "gradient_checkpointing": True,
    "dataloader_num_workers": 4,       # keep moderate to avoid CPU memory pressure
    "dataloader_pin_memory": True,
    "dataloader_prefetch_factor": 2,
    "torch_compile": False,            # unstable with multi-GPU DDP/FSDP in torch 2.2

    # Saving
    "save_steps": 500,
    "save_total_limit": 3,
    "logging_steps": 10,

    # Monitoring
    "report_to": "none",
    "run_name": "imi-foundation-8xa100-full-ft",
}


# =============================================================================
# DATA LOADING
# Handles both data formats produced by prepare_medical_data.py:
#   general_knowledge_{train,val}.json  — {"text": "..."}
#   instruction_{train,val}.json        — {"instruction":"...", "input":"...", "output":"..."}
# =============================================================================

def _format_example(ex: Dict) -> Optional[str]:
    """
    Convert a single example to Mixtral chat-formatted text.

    general_knowledge: raw text wrapped in <s>...</s>
    instruction:       Mixtral [INST] / [/INST] template
    """
    # general_knowledge format — has only 'text'
    if "text" in ex and "instruction" not in ex:
        text = ex["text"].strip()
        if text:
            return f"<s>{text}</s>"
        return None

    # instruction format — has 'instruction' + 'output'
    instruction = ex.get("instruction", "").strip()
    inp = ex.get("input", "").strip()
    output = ex.get("output", "").strip()

    if not instruction or not output:
        return None

    user_msg = f"{instruction}\n\n{inp}".strip() if inp else instruction
    return f"<s>[INST] {user_msg} [/INST] {output}</s>"


def _stable_hash(text: str) -> str:
    """Deterministic hash for deduplication (unlike Python's built-in hash())."""
    return hashlib.sha256(text.encode()).hexdigest()


def load_foundation_dataset(data_dir: Path, max_examples: Optional[int] = None) -> Dataset:
    """Load and combine all processed datasets for foundation training."""
    logger.info("Loading foundation training data...")

    all_examples = []
    final_dir = data_dir / "final"
    train_dir = data_dir / "train"
    processed_dir = data_dir / "processed"

    # Prefer final/ — output of prepare_medical_data.py
    if final_dir.exists():
        for f in sorted(final_dir.glob("*_train.json")):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                all_examples.extend(data)
                logger.info(f"  Loaded {len(data):,} from final/{f.name}")
            except Exception as e:
                logger.warning(f"  Failed to load {f.name}: {e}")

    # Fallback: train/ directory
    if not all_examples and train_dir.exists():
        for f in sorted(train_dir.glob("*_train.json")):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                all_examples.extend(data)
                logger.info(f"  Loaded {len(data):,} from train/{f.name}")
            except Exception as e:
                logger.warning(f"  Failed to load {f.name}: {e}")

    # Fallback: processed/ subdirectories
    if not all_examples and processed_dir.exists():
        for adapter_dir in sorted(processed_dir.iterdir()):
            if adapter_dir.is_dir():
                for f in sorted(adapter_dir.glob("*.json")):
                    try:
                        with open(f) as fp:
                            data = json.load(fp)
                        all_examples.extend(data)
                        logger.info(f"  Loaded {len(data):,} from {adapter_dir.name}/{f.name}")
                    except Exception as e:
                        logger.warning(f"  Failed to load {f.name}: {e}")

    if not all_examples:
        raise ValueError(
            "No training data found!\n"
            "  Run: python scripts/data_collection/collect_datasets.py\n"
            "  Then: python scripts/training/prepare_medical_data.py"
        )

    # Format and deduplicate using stable hash
    seen = set()
    formatted = []
    for ex in all_examples:
        text = _format_example(ex)
        if not text:
            continue
        h = _stable_hash(text)
        if h not in seen:
            seen.add(h)
            formatted.append({"text": text})

    logger.info(f"Total: {len(all_examples):,} raw → {len(formatted):,} unique formatted")

    random.shuffle(formatted)

    if max_examples and len(formatted) > max_examples:
        formatted = formatted[:max_examples]
        logger.info(f"Capped to {max_examples:,} examples")

    return Dataset.from_list(formatted)


# =============================================================================
# ZeRO STAGE 3 WEIGHT LOADING
# =============================================================================

def _load_zero3_pretrained_weights(model: torch.nn.Module, model_name: str, dtype: torch.dtype) -> None:
    """Load HuggingFace pretrained weights into a DeepSpeed ZeRO Stage 3 partitioned model.

    Cannot use from_pretrained inside zero.Init because:
    - zero.Init immediately partitions params to size [0] on non-owning ranks
    - from_pretrained's weight-copy then fails with shape mismatch
    Instead: create empty model with from_config inside zero.Init, then load
    weights shard-by-shard here using GatheredParameters so each param is
    temporarily gathered on rank 0 for the copy, then automatically re-partitioned.
    """
    import deepspeed

    # Resolve model_name to a local directory (HuggingFace cache or local path)
    local_path = Path(model_name)
    if not local_path.is_dir():
        from transformers.utils import cached_file
        _idx = cached_file(model_name, "model.safetensors.index.json", _raise_exceptions_for_missing_entries=False)
        if _idx is None:
            _idx = cached_file(model_name, "pytorch_model.bin.index.json", _raise_exceptions_for_missing_entries=False)
        if _idx is None:
            raise FileNotFoundError(f"Cannot find weight index for {model_name}")
        local_path = Path(_idx).parent

    # Build weight_map: param_name → shard filename
    index_file = local_path / "model.safetensors.index.json"
    if not index_file.exists():
        index_file = local_path / "pytorch_model.bin.index.json"
    if index_file.exists():
        with open(index_file) as f:
            weight_map: Dict[str, str] = json.load(f)["weight_map"]
    else:
        # Single-file model
        sf = local_path / "model.safetensors"
        pt = local_path / "pytorch_model.bin"
        fname = sf.name if sf.exists() else pt.name
        if sf.exists():
            from safetensors.torch import load_file as _sf_load
            weight_map = {k: fname for k in _sf_load(str(sf), device="cpu")}
        else:
            weight_map = {k: fname for k in torch.load(str(pt), map_location="cpu")}

    # Group params by shard file so we load each file only once
    shard_to_params: Dict[str, list] = defaultdict(list)
    param_dict = dict(model.named_parameters())
    for pname in param_dict:
        shard = weight_map.get(pname)
        if shard:
            shard_to_params[shard].append(pname)

    logger.info(f"Loading weights from {len(shard_to_params)} shard(s) into ZeRO3 model...")
    for shard_filename, param_names in shard_to_params.items():
        params_to_gather = [param_dict[n] for n in param_names]
        # GatheredParameters temporarily restores full tensors on modifier_rank=0,
        # lets us copy checkpoint data in, then re-partitions on __exit__.
        with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
            if dist.get_rank() == 0:
                shard_path = local_path / shard_filename
                if shard_path.suffix == ".safetensors":
                    from safetensors.torch import load_file as _sf_load
                    shard_sd = _sf_load(str(shard_path), device="cpu")
                else:
                    shard_sd = torch.load(str(shard_path), map_location="cpu")
                for pname in param_names:
                    if pname in shard_sd:
                        param_dict[pname].data.copy_(shard_sd[pname].to(dtype))
                del shard_sd

    dist.barrier()
    logger.info("Pretrained weights loaded into ZeRO3 model.")


# =============================================================================
# TRAINING
# =============================================================================

def train_foundation(
    base_model: str = None,
    data_dir: str = None,
    output_dir: str = None,
    max_examples: Optional[int] = None,
    resume_from: Optional[str] = None,
    deepspeed_config: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
):
    """Run full fine-tuning on 8× A100 80GB with DeepSpeed ZeRO Stage 3."""
    config = dict(A100_8GPU_CONFIG)
    if config_overrides:
        config.update(config_overrides)

    base_model = base_model or config["base_model"]
    data_path = Path(data_dir) if data_dir else DATA_DIR
    output_path = output_dir or str(MODELS_DIR / "foundation")

    logger.info("=" * 60)
    logger.info("IMI Foundation Training — Full Fine-Tuning on 8× A100 80GB")
    logger.info("=" * 60)
    logger.info(f"Base model:         {base_model}")
    logger.info(f"Data dir:           {data_path}")
    logger.info(f"Output:             {output_path}")
    logger.info(f"Epochs:             {config['num_epochs']}")
    logger.info(f"Per-device batch:   {config['batch_size']}")
    logger.info(f"Grad accumulation:  {config['gradient_accumulation_steps']}")
    logger.info(f"Seq length:         {config['max_seq_length']}")
    logger.info(f"Packing:            {config['packing']}")
    logger.info(f"Grad checkpointing: {config['gradient_checkpointing']}")
    logger.info(f"Optimizer:          {config['optim']}")
    logger.info(f"DeepSpeed config:   {deepspeed_config or 'NOT SET — will OOM without ZeRO'}")

    if not deepspeed_config:
        logger.warning(
            "No DeepSpeed config provided! Without ZeRO Stage 3, full fine-tuning "
            "of Mixtral 8x7B WILL run out of memory on 8× A100 80GB.\n"
            "  Pass: --deepspeed configs/deepspeed_zero3.json"
        )

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU:                {gpu_count}× {gpu_name} ({gpu_mem:.0f} GB each)")
        logger.info(f"Total VRAM:         {gpu_count * gpu_mem:.0f} GB combined")
    else:
        logger.error("No CUDA GPU detected!")
        return

    # TF32 for A100 matmul acceleration
    if config.get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load dataset
    dataset = load_foundation_dataset(data_path, max_examples)
    logger.info(f"Training on {len(dataset):,} examples")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = config["max_seq_length"]

    # Flash Attention 2 — with eager fallback
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        logger.info("Attention: Flash Attention 2")
    except ImportError:
        attn_impl = "eager"
        logger.info("Attention: eager (install flash-attn for ~30% speedup)")

    # Load model — FULL BF16, no quantization
    if deepspeed_config:
        import deepspeed
        with open(deepspeed_config) as _f:
            _ds_cfg = json.load(_f)
        # ZeRO Stage 3 loading — two-step to avoid shape-mismatch and meta-tensor errors:
        #   Step 1: create empty partitioned model structure inside zero.Init()
        #           (parameters are immediately sharded across GPUs during __init__)
        #   Step 2: load pretrained weights shard-by-shard using GatheredParameters
        #           (rank 0 reads each shard file and copies into gathered params;
        #            zero.Init re-partitions on context exit; other ranks sync via barrier)
        logger.info("ZeRO3 Step 1: creating empty partitioned model from config...")
        _model_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        with deepspeed.zero.Init(config_dict_or_path=_ds_cfg, dtype=torch.bfloat16):
            model = AutoModelForCausalLM.from_config(
                _model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
        logger.info("ZeRO3 Step 2: loading pretrained weights into partitioned model...")
        _load_zero3_pretrained_weights(model, base_model, torch.bfloat16)
    else:
        logger.info("Loading model in BFloat16 (full parameters, no quantization)...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=None,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )

    # Gradient checkpointing — mandatory for A100 full FT to avoid OOM
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} (100% — full fine-tuning)")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        max_grad_norm=config["max_grad_norm"],
        optim=config["optim"],                 # adamw_torch — compatible with ZeRO
        fp16=config["fp16"],
        bf16=config["bf16"],
        tf32=config.get("tf32", True),
        logging_steps=config["logging_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        report_to=config["report_to"],
        run_name=config["run_name"],
        dataloader_num_workers=config.get("dataloader_num_workers", 4),
        dataloader_pin_memory=config.get("dataloader_pin_memory", True),
        dataloader_prefetch_factor=config.get("dataloader_prefetch_factor", 2),
        group_by_length=True,
        gradient_checkpointing=config["gradient_checkpointing"],
        ddp_find_unused_parameters=False,
        torch_compile=config.get("torch_compile", False),  # disabled for multi-GPU stability
        deepspeed=deepspeed_config,            # ZeRO Stage 3 config path (None = no ZeRO)
        remove_unused_columns=False,
    )

    # SFT Trainer — pass packing and text field explicitly
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",             # explicit — avoids trl warning/error
        max_seq_length=config["max_seq_length"],
        packing=config["packing"],             # was in config but never passed before
    )

    # Train
    logger.info("Starting full fine-tuning...")
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Save — on multi-GPU, only rank 0 saves
    logger.info(f"Saving foundation model to: {output_path}")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    metadata = {
        "training_type": "full_fine_tuning",
        "training_hardware": "8x A100 80GB",
        "base_model": base_model,
        "num_examples": len(dataset),
        "config": config,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "precision": "bfloat16",
        "deepspeed": deepspeed_config,
    }
    with open(Path(output_path) / "foundation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 60)
    logger.info("Foundation training complete!")
    logger.info(f"Next step: python scripts/training/finetune_mixtral.py \\")
    logger.info(f"    --base-model {output_path} --gpu-tier A100_80GB")
    logger.info("=" * 60)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Foundation Training — Full Fine-Tuning on 8× A100 80GB"
    )
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-packing", action="store_true")
    parser.add_argument(
        "--deepspeed", default=None,
        help="Path to DeepSpeed config JSON (strongly recommended: configs/deepspeed_zero3.json)"
    )

    args = parser.parse_args()

    overrides = {}
    if args.epochs:
        overrides["num_epochs"] = args.epochs
    if args.lr:
        overrides["learning_rate"] = args.lr
    if args.batch_size:
        overrides["batch_size"] = args.batch_size    # fixed key name (was per_device_batch_size)
    if args.no_packing:
        overrides["packing"] = False

    train_foundation(
        base_model=args.base_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        resume_from=args.resume_from,
        deepspeed_config=args.deepspeed,       # now actually forwarded
        config_overrides=overrides if overrides else None,
    )


if __name__ == "__main__":
    main()
