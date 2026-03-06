#!/usr/bin/env bash
# =============================================================================
# install_training.sh — Safe CUDA-aware training environment setup
# =============================================================================
#
# Installs the Mixtral 8x7B training stack in the correct order.
# Covers both:
#   (A) Foundation training  — full fine-tuning, 8× A100 80GB, DeepSpeed ZeRO-3
#   (B) Adapter training     — QLoRA, single A100 80GB
#
# Installing packages in the wrong order (or all at once) will break CUDA,
# corrupt your torch install, or produce silent compute failures.
#
# Usage:
#   bash scripts/install_training.sh                    # adapter training only
#   bash scripts/install_training.sh --foundation       # include DeepSpeed for foundation training
#   bash scripts/install_training.sh --cuda 12.1        # force CUDA 12.1
#   bash scripts/install_training.sh --cuda 11.8        # force CUDA 11.8
#   bash scripts/install_training.sh --skip-flash       # skip flash-attention build
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
INSTALL_DEEPSPEED=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cuda) CUDA_VERSION="$2"; shift ;;
        --skip-flash) SKIP_FLASH_ATTN=true ;;
        --skip-wandb) SKIP_WANDB=true ;;
        --foundation) INSTALL_DEEPSPEED=true ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

echo "============================================================"
echo "  IMI Mixtral 8x7B — Training Environment Setup"
if [[ "$INSTALL_DEEPSPEED" == true ]]; then
    echo "  Mode: Foundation Training (includes DeepSpeed ZeRO-3)"
else
    echo "  Mode: Adapter Training / QLoRA (no DeepSpeed)"
    echo "  For foundation training on 8xA100, add --foundation flag"
fi
echo "============================================================"

# ── Step 0: Verify Python version ───────────────────────────────────────────
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "[0/7] Python: $PYTHON_VERSION"
if [[ "$PYTHON_VERSION" == "3.12" ]]; then
    echo "  WARNING: Python 3.12 has known issues with bitsandbytes."
    echo "  Strongly recommend Python 3.10 or 3.11."
    read -p "  Continue anyway? [y/N]: " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

# ── Step 1: Detect CUDA version ─────────────────────────────────────────────
echo "[1/7] Detecting CUDA version..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "  ERROR: nvidia-smi not found. Is an NVIDIA GPU present?"
    exit 1
fi

# ── Step 2: Install or reuse PyTorch ────────────────────────────────────────
echo ""
echo "[2/7] Checking for existing PyTorch installation..."
echo "  (RunPod pre-installs torch+torchaudio+torchvision as a matched set."
echo "   Reinstalling torch alone breaks torchaudio/torchvision.)"

TORCH_PREINSTALLED=false
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    EXISTING_TORCH=$(python -c "import torch; print(torch.__version__)")
    CUDA_TAG=$(python -c "
import torch
v = torch.version.cuda  # e.g. '12.4'
tag = 'cu' + v.replace('.', '')  # e.g. 'cu124'
print(tag)
")
    echo "  Found pre-installed torch $EXISTING_TORCH with CUDA tag $CUDA_TAG"
    echo "  CUDA is available — skipping torch install to preserve torchaudio/torchvision."
    TORCH_PREINSTALLED=true
else
    echo "  No working torch+CUDA found. Installing fresh..."

    # Determine CUDA tag
    if [[ "$CUDA_VERSION" != "auto" ]]; then
        # Manual override: "12.4" -> "cu124", "11.8" -> "cu118"
        CUDA_TAG="cu$(echo $CUDA_VERSION | tr -d '.')"
    elif nvcc --version 2>/dev/null | grep -q "12\.4"; then
        CUDA_TAG="cu124"
    elif nvcc --version 2>/dev/null | grep -q "12\.1\|12\.2\|12\.3"; then
        CUDA_TAG="cu121"
    elif nvcc --version 2>/dev/null | grep -q "11\.8"; then
        CUDA_TAG="cu118"
    else
        echo "  WARNING: Could not detect CUDA version — defaulting to cu124 (RunPod default)."
        echo "  Override with: --cuda 12.1 or --cuda 11.8"
        CUDA_TAG="cu124"
    fi

    # Select torch version for the CUDA tag
    if [[ "$CUDA_TAG" == "cu124" ]]; then
        TORCH_VER="2.4.1"
    else
        TORCH_VER="2.2.0"
    fi

    TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
    echo "  Installing torch==${TORCH_VER}+${CUDA_TAG} from $TORCH_INDEX"

    pip install \
        "torch==${TORCH_VER}" \
        --index-url "$TORCH_INDEX" \
        --upgrade
fi

# Verify CUDA works
python -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU count: {torch.cuda.device_count()}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
    echo "  ERROR: CUDA is not available."
    echo "  Check your CUDA driver or override with --cuda <version>."
    exit 1
fi

# Extract CUDA tag from the installed torch (covers both paths above)
CUDA_TAG=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))")
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
echo "  CUDA tag: $CUDA_TAG  |  Index: $TORCH_INDEX"

# ── Step 3: Install core training stack ─────────────────────────────────────
echo ""
echo "[3/7] Installing core training stack..."
echo "  (transformers, accelerate, peft, trl, bitsandbytes, datasets)"

pip install --upgrade \
    "transformers==4.44.2" \
    "accelerate==0.34.2" \
    "peft==0.12.0" \
    "trl==0.9.6" \
    "bitsandbytes==0.44.0" \
    "datasets==2.21.0" \
    "safetensors==0.4.5"

# ── Step 4: Install tokenization & utility packages ──────────────────────────
echo ""
echo "[4/7] Installing tokenization and utility packages..."

pip install --upgrade \
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
    echo "[5/7] Skipping flash-attention (--skip-flash specified)."
    echo "  Training will work without it (~30-50% slower on attention ops)."
else
    echo "[5/7] Installing flash-attention 2 (builds from source, may take 5-15 min)..."
    echo "  This speeds up training by ~30-50% on A100/H100."
    echo "  If this fails, re-run with --skip-flash and training will still work."

    pip install "flash-attn==2.5.9" --no-build-isolation || {
        echo "  WARNING: flash-attn install failed. Training will continue without it."
        echo "  The training script will automatically disable it."
    }
fi

# ── Step 6: DeepSpeed — required for foundation training on 8× A100 ──────────
echo ""
if [[ "$INSTALL_DEEPSPEED" == true ]]; then
    echo "[6/7] Installing DeepSpeed 0.14.0 (foundation training, 8× A100)..."
    echo "  This builds CUDA kernels — may take 5–10 minutes."
    echo "  Requires gcc, g++, ninja. Installing ninja first..."

    pip install "ninja>=1.11.1" "packaging>=23.0"

    # DeepSpeed install — build CPU Adam op at minimum
    DS_BUILD_CPU_ADAM=1 pip install "deepspeed==0.14.0" || {
        echo "  WARNING: deepspeed install failed."
        echo "  Try: apt-get install -y ninja-build build-essential"
        echo "  Then re-run with --foundation"
    }

    # Verify
    python -c "
import deepspeed
print(f'deepspeed:      {deepspeed.__version__}')
" || echo "  deepspeed: NOT INSTALLED"

    ds_report 2>/dev/null | grep -E "op_name|available" | head -20 || true
else
    echo "[6/7] Skipping DeepSpeed (adapter training mode)."
    echo "  For foundation training on 8× A100, re-run with --foundation"
    pip install "ninja>=1.11.1" "packaging>=23.0"
fi

# ── Step 7: Optional experiment tracking ─────────────────────────────────────
echo ""
if [[ "$SKIP_WANDB" == true ]]; then
    echo "[7/7] Skipping wandb (--skip-wandb specified)."
else
    echo "[7/7] Installing wandb for experiment tracking (optional)..."
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

try:
    import deepspeed
    print(f'deepspeed:      {deepspeed.__version__}')
except:
    print('deepspeed:      not installed (needed only for foundation training)')
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
if [[ "$INSTALL_DEEPSPEED" == true ]]; then
echo "  3. Demo test (single GPU, verify pipeline):"
echo "     python scripts/training/finetune_mixtral.py --demo"
echo ""
echo "  4. Foundation training (8× A100 80GB):"
echo "     torchrun --nproc_per_node=8 scripts/training/train_foundation.py \\"
echo "         --deepspeed configs/deepspeed_zero3.json"
echo ""
echo "  5. Adapter training (single A100 80GB, after foundation):"
echo "     python scripts/training/finetune_mixtral.py \\"
echo "         --base-model models/foundation --gpu-tier A100_80GB"
else
echo "  3. Demo test (100 examples, verify everything works):"
echo "     python scripts/training/finetune_mixtral.py --demo"
echo ""
echo "  4. Adapter training:"
echo "     python scripts/training/finetune_mixtral.py --gpu-tier A100_80GB"
fi
echo "============================================================"
