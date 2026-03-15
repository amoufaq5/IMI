"""
Checkpoint Averaging for IMI Medical LLM

Averages the weights of the last N checkpoints to produce a single model
that typically outperforms any individual checkpoint with zero extra training.

Why it works:
  - Each checkpoint sits at a different point in the loss landscape.
  - Averaging moves towards a flatter, wider minimum with better generalization.
  - Particularly effective after cosine-decay schedules (last ~10% of training).

Usage:
    # Average last 3 checkpoints from foundation training
    python scripts/training/average_checkpoints.py \\
        --checkpoint-dir models/foundation \\
        --output-dir models/foundation_averaged \\
        --num-checkpoints 3

    # Average last 5 adapter checkpoints
    python scripts/training/average_checkpoints.py \\
        --checkpoint-dir models/mixtral-medical-qlora \\
        --output-dir models/mixtral-medical-qlora-averaged \\
        --num-checkpoints 5

Output:
    models/{output_dir}/  — merged model weights + tokenizer
    models/{output_dir}/averaging_metadata.json — which checkpoints were merged
"""
import json
import logging
import argparse
import shutil
from pathlib import Path
from typing import List, Optional

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def find_checkpoints(checkpoint_dir: Path, num_checkpoints: int) -> List[Path]:
    """Find the last N checkpoints sorted by step number."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Collect checkpoint-NNNN subdirectories
    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
    )

    if not checkpoints:
        # The directory itself might be a checkpoint (no subdirectories)
        # In that case average is a no-op — just copy the model
        logger.warning(f"No checkpoint-NNNN subdirectories found in {checkpoint_dir}. "
                       "Treating the directory itself as a single checkpoint.")
        return [checkpoint_dir]

    selected = checkpoints[-num_checkpoints:]
    logger.info(f"Found {len(checkpoints)} checkpoints, selected last {len(selected)}:")
    for ckpt in selected:
        logger.info(f"  {ckpt}")
    return selected


def average_safetensors(checkpoints: List[Path], output_dir: Path) -> None:
    """Average model weights stored as safetensors shards."""
    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        raise ImportError("safetensors not installed. Run: pip install safetensors")

    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(checkpoints)

    # Collect shard filenames from first checkpoint
    first = checkpoints[0]
    shard_files = sorted(first.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files found in {first}")

    logger.info(f"Averaging {n} checkpoints across {len(shard_files)} shard(s)...")

    for shard_file in shard_files:
        shard_name = shard_file.name
        logger.info(f"  Processing shard: {shard_name}")

        # Load and accumulate
        accumulated: dict = {}
        for ckpt in checkpoints:
            shard_path = ckpt / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard {shard_name} missing from checkpoint {ckpt}")
            tensors = load_file(str(shard_path))
            for key, tensor in tensors.items():
                if key in accumulated:
                    accumulated[key] = accumulated[key] + tensor.float()
                else:
                    accumulated[key] = tensor.float()

        # Divide by n to get mean
        averaged = {key: (val / n).to(torch.bfloat16) for key, val in accumulated.items()}

        # Save averaged shard
        out_path = output_dir / shard_name
        save_file(averaged, str(out_path))
        logger.info(f"  Saved averaged shard: {out_path}")


def copy_config_files(source: Path, output_dir: Path) -> None:
    """Copy non-weight config files (tokenizer, model config, etc.)."""
    config_patterns = [
        "config.json",
        "tokenizer*.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "tokenizer.model",
        "generation_config.json",
        "model.safetensors.index.json",
    ]
    for pattern in config_patterns:
        for f in source.glob(pattern):
            dest = output_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
                logger.info(f"  Copied: {f.name}")


def average_checkpoints(
    checkpoint_dir: str,
    output_dir: Optional[str] = None,
    num_checkpoints: int = 3,
) -> str:
    """Average the last N checkpoints and save the result.

    Args:
        checkpoint_dir: Directory containing checkpoint-NNNN subdirectories.
        output_dir: Where to save the averaged model. Defaults to
                    {checkpoint_dir}_averaged.
        num_checkpoints: Number of checkpoints to average (default: 3).

    Returns:
        Path to the averaged model directory.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if output_dir is None:
        output_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}_averaged"
    output_dir = Path(output_dir)

    logger.info("=" * 60)
    logger.info("Checkpoint Averaging")
    logger.info("=" * 60)
    logger.info(f"Source: {checkpoint_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Checkpoints to average: {num_checkpoints}")

    checkpoints = find_checkpoints(checkpoint_dir, num_checkpoints)

    if len(checkpoints) == 1:
        logger.info("Only one checkpoint found — copying without averaging.")
        if output_dir != checkpoints[0]:
            shutil.copytree(checkpoints[0], output_dir, dirs_exist_ok=True)
        return str(output_dir)

    # Average weights
    average_safetensors(checkpoints, output_dir)

    # Copy config files from the latest checkpoint
    copy_config_files(checkpoints[-1], output_dir)

    # Save metadata
    metadata = {
        "averaging_method": "uniform_mean",
        "num_checkpoints_averaged": len(checkpoints),
        "source_checkpoints": [str(c) for c in checkpoints],
        "output_dtype": "bfloat16",
    }
    with open(output_dir / "averaging_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Averaged model saved to: {output_dir}")
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Average last N training checkpoints")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "foundation"),
        help="Directory containing checkpoint-NNNN subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for averaged model (default: {checkpoint_dir}_averaged)",
    )
    parser.add_argument(
        "--num-checkpoints",
        type=int,
        default=3,
        help="Number of most recent checkpoints to average (default: 3)",
    )
    args = parser.parse_args()

    average_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        num_checkpoints=args.num_checkpoints,
    )


if __name__ == "__main__":
    main()
