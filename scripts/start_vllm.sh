#!/bin/bash
# ============================================================================
# IMI Medical AI — vLLM Inference Server with LoRA Hot-Swapping
#
# Serves Mistral 7B for inference.
# Adapters are loaded at startup and hot-swapped per request based on user_type.
#
# Usage:
#   bash scripts/start_vllm.sh                    # All adapters
#   bash scripts/start_vllm.sh --mvp              # Doctor + Patient only
#   bash scripts/start_vllm.sh --adapters doctor patient  # Custom selection
#
# Requirements:
#   - 1x A100 80GB GPU minimum (2x recommended for production)
#   - vLLM installed: pip install vllm
#   - DPO-aligned model at ./models/dpo_aligned/
#   - Adapter weights at ./models/adapters/{type}_adapter/
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Defaults
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/dpo_aligned}"
ADAPTERS_DIR="${MODEL_PATH}/../adapters"
PORT="${PORT:-8080}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# All 6 adapter definitions
declare -A ADAPTER_PATHS=(
    ["student"]="${ADAPTERS_DIR}/education_adapter"
    ["doctor"]="${ADAPTERS_DIR}/clinical_decision_adapter"
    ["patient"]="${ADAPTERS_DIR}/patient_triage_adapter"
    ["researcher"]="${ADAPTERS_DIR}/research_adapter"
    ["hospital"]="${ADAPTERS_DIR}/clinical_decision_adapter"
    ["pharma"]="${ADAPTERS_DIR}/regulatory_qa_adapter"
    ["pharmacist"]="${ADAPTERS_DIR}/clinical_pharmacist_adapter"
)

# Parse arguments
SELECTED_ADAPTERS=()
MVP_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mvp)
            MVP_MODE=true
            SELECTED_ADAPTERS=("doctor" "patient")
            shift
            ;;
        --adapters)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                SELECTED_ADAPTERS+=("$1")
                shift
            done
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --tp)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--mvp] [--adapters name1 name2 ...] [--port PORT] [--tp N] [--model PATH]"
            echo ""
            echo "Options:"
            echo "  --mvp              Load only doctor + patient adapters (saves VRAM)"
            echo "  --adapters NAMES   Load specific adapters"
            echo "  --port PORT        Server port (default: 8080)"
            echo "  --tp N             Tensor parallel size (default: 2)"
            echo "  --model PATH       Path to base/DPO model"
            echo ""
            echo "Available adapters: ${!ADAPTER_PATHS[*]}"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default: load all adapters
if [ ${#SELECTED_ADAPTERS[@]} -eq 0 ]; then
    SELECTED_ADAPTERS=("student" "doctor" "patient" "researcher" "hospital" "pharma" "pharmacist")
fi

# Build --lora-modules argument
LORA_MODULES=""
LOADED_COUNT=0
for adapter in "${SELECTED_ADAPTERS[@]}"; do
    path="${ADAPTER_PATHS[$adapter]:-}"
    if [ -z "$path" ]; then
        echo "WARNING: Unknown adapter '$adapter', skipping"
        continue
    fi
    if [ ! -d "$path" ]; then
        echo "WARNING: Adapter path not found: $path (adapter: $adapter), skipping"
        continue
    fi
    if [ -n "$LORA_MODULES" ]; then
        LORA_MODULES="${LORA_MODULES} "
    fi
    LORA_MODULES="${LORA_MODULES}${adapter}=${path}"
    LOADED_COUNT=$((LOADED_COUNT + 1))
done

echo "============================================================"
echo "IMI Medical AI — vLLM Inference Server"
echo "============================================================"
echo "Model:           ${MODEL_PATH}"
echo "Port:            ${PORT}"
echo "Tensor Parallel: ${TENSOR_PARALLEL}"
echo "Max Model Len:   ${MAX_MODEL_LEN}"
echo "GPU Memory Util: ${GPU_MEMORY_UTILIZATION}"
echo "Adapters (${LOADED_COUNT}):    ${SELECTED_ADAPTERS[*]}"
if $MVP_MODE; then
    echo "Mode:            MVP (Doctor + Patient only)"
fi
echo "============================================================"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo ""
    echo "ERROR: Model not found at ${MODEL_PATH}"
    echo ""
    echo "Training pipeline order:"
    echo "  1. python scripts/training/train_foundation.py"
    echo "  2. python scripts/training/train_dpo.py train"
    echo "  3. python scripts/training/train_lora.py --adapter all"
    echo "  4. bash scripts/start_vllm.sh"
    exit 1
fi

# Launch vLLM
CMD="python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --tensor-parallel-size ${TENSOR_PARALLEL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --port ${PORT} \
    --trust-remote-code \
    --dtype bfloat16"

# Add LoRA modules if any adapters were found
if [ -n "$LORA_MODULES" ]; then
    CMD="${CMD} --enable-lora --lora-modules ${LORA_MODULES}"
fi

echo ""
echo "Starting vLLM server..."
echo "API endpoint: http://localhost:${PORT}/v1"
echo "Health check: curl http://localhost:${PORT}/health"
echo ""

exec $CMD
