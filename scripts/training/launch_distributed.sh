#!/bin/bash
# UMI Distributed Training Launch Script
# Optimized for 4xH100 GPUs

set -e

# Configuration
NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29500}
OUTPUT_DIR=${OUTPUT_DIR:-"models/umi-medical-v2"}
MODE=${MODE:-"full"}  # "full" or "qlora"

# Model settings
BASE_MODEL=${BASE_MODEL:-"mistralai/Mistral-7B-Instruct-v0.2"}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-8192}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-2e-5}

# DeepSpeed settings
DS_STAGE=${DS_STAGE:-2}

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
TRAIN_SCRIPT="$SCRIPT_DIR/fine_tune_distributed.py"

echo "=============================================="
echo "UMI Distributed Training"
echo "=============================================="
echo "GPUs: $NUM_GPUS"
echo "Mode: $MODE"
echo "Model: $BASE_MODEL"
echo "Sequence Length: $MAX_SEQ_LENGTH"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE x $GRAD_ACCUM (grad accum)"
echo "Effective Batch: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "Learning Rate: $LR"
echo "DeepSpeed Stage: $DS_STAGE"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA not available."
    exit 1
fi

echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Activate virtual environment if exists
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Set environment variables for optimal H100 performance
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export TOKENIZERS_PARALLELISM=false

# Enable TF32 for H100
export NVIDIA_TF32_OVERRIDE=1

# Run with DeepSpeed
if [ "$MODE" == "full" ]; then
    echo "Starting FULL fine-tune with DeepSpeed ZeRO Stage $DS_STAGE..."
    
    deepspeed --num_gpus=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        "$TRAIN_SCRIPT" \
        --base-model "$BASE_MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --mode full \
        --train-data-dir "$PROJECT_ROOT/data/training" \
        --knowledge-base-dir "$PROJECT_ROOT/data/knowledge_base" \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --grad-accum $GRAD_ACCUM \
        --lr $LR \
        --max-seq-length $MAX_SEQ_LENGTH \
        --deepspeed \
        --deepspeed-stage $DS_STAGE

elif [ "$MODE" == "qlora" ]; then
    echo "Starting QLoRA fine-tune..."
    
    deepspeed --num_gpus=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        "$TRAIN_SCRIPT" \
        --base-model "$BASE_MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --mode qlora \
        --train-data-dir "$PROJECT_ROOT/data/training" \
        --knowledge-base-dir "$PROJECT_ROOT/data/knowledge_base" \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --grad-accum $GRAD_ACCUM \
        --lr 2e-4 \
        --max-seq-length $MAX_SEQ_LENGTH \
        --deepspeed \
        --deepspeed-stage $DS_STAGE \
        --merge

else
    echo "ERROR: Unknown mode '$MODE'. Use 'full' or 'qlora'."
    exit 1
fi

echo ""
echo "=============================================="
echo "Training Complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "=============================================="
