#!/usr/bin/env bash
# =============================================================================
# install_training.sh — Safe CUDA-aware training environment setup
# =============================================================================
#
# Installs the Mixtral 8x7B QLoRA training stack in the correct order.
# Installing packages in the wrong order (or all at once) will break CUDA,
# corrupt your torch install, or produce silent compute failures.
#
# Usage:
#   bash scripts/install_training.sh              # auto-detect CUDA version
#   bash scripts/install_training.sh --cuda 12.1  # force CUDA 12.1
#   bash scripts/install_training.sh --cuda 11.8  # force CUDA 11.8
#   bash scripts/install_training.sh --skip-flash # skip flash-attention build
#
# Prerequisites:
#   - Python 3.10 or 3.11 (3.12 has bitsandbytes issues)
#   - NVIDIA GPU with CUDA driver installed
#   - nvidia-smi working in your shell
# =============================================================================

set -e  # Exit on first error

CUDA_VERSION="auto"
SKIP_FLASH_ATTN=false
SKIP_WANDB=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cuda) CUDA_VERSION="$2"; shift ;;
        --skip-flash) SKIP_FLASH_ATTN=true ;;
        --skip-wandb) SKIP_WANDB=true ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

echo "============================================================"
echo "  IMI Mixtral 8x7B — Training Environment Setup"
echo "============================================================"

# ── Step 0: Verify Python version ───────────────────────────────────────────
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "[0/6] Python: $PYTHON_VERSION"
if [[ "$PYTHON_VERSION" == "3.12" ]]; then
    echo "  WARNING: Python 3.12 has known issues with bitsandbytes."
    echo "  Strongly recommend Python 3.10 or 3.11."
    read -p "  Continue anyway? [y/N]: " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

# ── Step 1: Detect CUDA version ─────────────────────────────────────────────
echo "[1/6] Detecting CUDA version..."
if [[ "$CUDA_VERSION" == "auto" ]]; then
    if ! command -v nvidia-smi &> /dev/null; then
        echo "  ERROR: nvidia-smi not found. Is an NVIDIA GPU present?"
        exit 1
    fi
    DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    NVCC_CUDA=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | cut -d',' -f1 || echo "unknown")
    echo "  GPU Driver CUDA: $DRIVER_CUDA"
    echo "  NVCC CUDA: $NVCC_CUDA"

    # Determine torch CUDA tag
    if nvcc --version 2>/dev/null | grep -q "12\.1\|12\.2\|12\.3\|12\.4"; then
        CUDA_TAG="cu121"
    elif nvcc --version 2>/dev/null | grep -q "12\."; then
        CUDA_TAG="cu121"  # Use 12.1 wheels for any CUDA 12.x
    elif nvcc --version 2>/dev/null | grep -q "11\.8"; then
        CUDA_TAG="cu118"
    else
        echo "  WARNING: Could not determine CUDA version from nvcc."
        echo "  Defaulting to CUDA 12.1 wheels. Override with --cuda 11.8 if needed."
        CUDA_TAG="cu121"
    fi
else
    # Manual override: accept "12.1", "11.8", "cu121", "cu118"
    CUDA_TAG="cu$(echo $CUDA_VERSION | tr -d '.')"
    # Normalize: "cu121" or "cu118"
    if [[ "$CUDA_VERSION" == "12."* ]]; then CUDA_TAG="cu121"; fi
    if [[ "$CUDA_VERSION" == "11.8" ]]; then CUDA_TAG="cu118"; fi
fi

TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
echo "  Using PyTorch index: $TORCH_INDEX"

# ── Step 2: Install PyTorch (FIRST — never upgrade this implicitly) ──────────
echo ""
echo "[2/6] Installing PyTorch 2.2.0 with CUDA ${CUDA_TAG}..."
echo "  This sets the CUDA baseline. All other packages must match."

pip install \
    "torch==2.2.0" \
    --index-url "$TORCH_INDEX" \
    --upgrade

# Verify CUDA is accessible
python -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
    echo "  ERROR: CUDA is not available after torch install."
    echo "  Check your CUDA driver and verify --cuda flag matches your system."
    exit 1
fi

# ── Step 3: Install core training stack ─────────────────────────────────────
echo ""
echo "[3/6] Installing core training stack..."
echo "  (transformers, accelerate, peft, trl, bitsandbytes, datasets)"

pip install \
    "transformers==4.42.0" \
    "accelerate==0.30.0" \
    "peft==0.11.1" \
    "trl==0.9.6" \
    "bitsandbytes==0.43.1" \
    "datasets==2.19.0" \
    "safetensors==0.4.3"

# ── Step 4: Install tokenization & utility packages ──────────────────────────
echo ""
echo "[4/6] Installing tokenization and utility packages..."

pip install \
    "sentencepiece==0.1.99" \
    "tokenizers==0.19.1" \
    "einops==0.8.0" \
    "numpy==1.26.4" \
    "pandas==2.2.1" \
    "tqdm>=4.65.0" \
    "pyyaml>=6.0.1"

# ── Step 5: Flash Attention 2 (optional but recommended) ─────────────────────
echo ""
if [[ "$SKIP_FLASH_ATTN" == true ]]; then
    echo "[5/6] Skipping flash-attention (--skip-flash specified)."
    echo "  Training will work without it (~30-50% slower on attention ops)."
else
    echo "[5/6] Installing flash-attention 2 (builds from source, may take 5-15 min)..."
    echo "  This speeds up training by ~30-50% on A100/H100."
    echo "  If this fails, re-run with --skip-flash and training will still work."

    pip install "flash-attn==2.5.9" --no-build-isolation || {
        echo "  WARNING: flash-attn install failed. Training will continue without it."
        echo "  The training script will automatically disable it."
    }
fi

# ── Step 6: Optional experiment tracking ─────────────────────────────────────
echo ""
if [[ "$SKIP_WANDB" == true ]]; then
    echo "[6/6] Skipping wandb (--skip-wandb specified)."
else
    echo "[6/6] Installing wandb for experiment tracking (optional)..."
    pip install "wandb>=0.16.0" || {
        echo "  WARNING: wandb install failed. Use --report-to none in training."
    }
fi

# ── Final verification ────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Verification"
echo "============================================================"
python -c "
import torch
print(f'torch:          {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

try:
    import transformers
    print(f'transformers:   {transformers.__version__}')
except: print('transformers:   NOT INSTALLED')

try:
    import peft
    print(f'peft:           {peft.__version__}')
except: print('peft:           NOT INSTALLED')

try:
    import trl
    print(f'trl:            {trl.__version__}')
except: print('trl:            NOT INSTALLED')

try:
    import bitsandbytes as bnb
    print(f'bitsandbytes:   {bnb.__version__}')
except: print('bitsandbytes:   NOT INSTALLED')

try:
    import datasets
    print(f'datasets:       {datasets.__version__}')
except: print('datasets:       NOT INSTALLED')

try:
    import flash_attn
    print(f'flash-attn:     {flash_attn.__version__}')
except:
    print('flash-attn:     not installed (optional)')
"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Collect data:"
echo "     python scripts/data_collection/collect_datasets.py"
echo ""
echo "  2. Prepare 2-format dataset:"
echo "     python scripts/training/prepare_medical_data.py"
echo ""
echo "  3. Demo test (100 examples, verify everything works):"
echo "     python scripts/training/finetune_mixtral.py --demo"
echo ""
echo "  4. Full training (replace A100_40GB with your GPU tier):"
echo "     python scripts/training/finetune_mixtral.py --gpu-tier A100_80GB"
echo "============================================================"
