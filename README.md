# IMI - Intelligent Medical Intelligence

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

> **Fine-tuned Medical LLM with Reinforcement Learning for Enhanced Reasoning**

## 🎯 Overview

IMI is a medical AI training pipeline that fine-tunes large language models on medical data with reinforcement learning for improved reasoning. The trained model powers three specialized applications:

- **💊 Pharma App**: Drug discovery, clinical trials, regulatory affairs
- **📚 Student App**: Medical education, USMLE prep, clinical reasoning
- **🏥 General App**: Health information for general users

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      IMI Training Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│  Data Ingestion    │  Data Processing   │  Training             │
│  - PubMed          │  - QA Generation   │  - SFT (Supervised)   │
│  - HuggingFace     │  - Deduplication   │  - DPO (Preference)   │
│  - Medical QA      │  - Format Convert  │  - ORPO (Combined)    │
├─────────────────────────────────────────────────────────────────┤
│                      Fine-tuned Medical LLM                      │
├─────────────────────────────────────────────────────────────────┤
│  Pharma App        │  Student App       │  General Health App   │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- CUDA 11.8+ (for GPU training)
- 24GB+ VRAM (for QLoRA) or 80GB+ (for full fine-tune)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/imi.git
cd imi

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install flash-attention (optional, for long sequences)
pip install flash-attn --no-build-isolation
```

### Full Training Pipeline

```bash
# Run complete pipeline: Ingest → Process → SFT → DPO
python scripts/training/run_training.py --all

# Or run individual stages:
python scripts/training/run_training.py --ingest    # Data ingestion
python scripts/training/run_training.py --process   # Data processing
python scripts/training/run_training.py --sft       # Supervised fine-tuning
python scripts/training/run_training.py --dpo       # Preference optimization

# Alternative: ORPO (combined SFT + preference, more efficient)
python scripts/training/run_training.py --orpo
```

### Launch Applications

```bash
# Pharma App (port 7860)
python apps/pharma/app.py --model outputs/imi-medical/sft

# Student App (port 7861)
python apps/student/app.py --model outputs/imi-medical/sft

# General Health App (port 7862)
python apps/general/app.py --model outputs/imi-medical/sft
```

## 📁 Project Structure

```
imi/
├── scripts/
│   ├── data_ingestion/          # Data scrapers
│   │   ├── base_scraper.py      # Base scraper class
│   │   ├── scrape_pubmed.py     # PubMed literature
│   │   ├── scrape_medical_datasets.py  # HuggingFace datasets
│   │   └── scrape_all.py        # Master ingestion script
│   │
│   └── training/                # Training pipeline
│       ├── data_processor.py    # Convert to training format
│       ├── sft_trainer.py       # Supervised fine-tuning
│       ├── dpo_trainer.py       # Direct Preference Optimization
│       ├── orpo_trainer.py      # Odds Ratio Preference Optimization
│       └── run_training.py      # Master training script
│
├── apps/                        # User applications
│   ├── pharma/app.py           # Pharmaceutical research assistant
│   ├── student/app.py          # Medical education assistant
│   └── general/app.py          # General health assistant
│
├── data/
│   ├── raw/                    # Scraped data
│   └── processed/              # Training-ready data
│
├── outputs/                    # Trained models
│
└── requirements.txt            # Dependencies
```

## 🔧 Training Configuration

### SFT (Supervised Fine-Tuning)

```bash
python scripts/training/sft_trainer.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --mode qlora \
    --max-seq-length 4096 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 2e-4 \
    --epochs 3 \
    --lora-r 64
```

### DPO (Direct Preference Optimization)

```bash
python scripts/training/dpo_trainer.py \
    --model outputs/imi-medical/sft \
    --beta 0.1 \
    --lr 5e-5 \
    --epochs 1
```

### ORPO (Combined SFT + Preference)

```bash
python scripts/training/orpo_trainer.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --beta 0.1 \
    --lr 8e-6 \
    --epochs 3
```

## 📊 Training Modes

| Mode | VRAM Required | Speed | Quality |
|------|---------------|-------|---------|
| QLoRA (4-bit) | 24GB | Fast | Good |
| LoRA (16-bit) | 48GB | Medium | Better |
| Full Fine-tune | 80GB+ | Slow | Best |

## 🧪 Data Sources

### Scraped Data
- **PubMed**: Medical literature and research articles
- **HuggingFace Datasets**:
  - PubMedQA: Research question answering
  - MedQA: USMLE-style questions
  - MedMCQA: Medical entrance exam questions
  - Medical Meadow: Curated medical QA
  - ChatDoctor: Doctor-patient conversations

### Training Formats
- **SFT**: Chat format with system prompts
- **DPO/ORPO**: Preference pairs (chosen/rejected responses)

## 🔬 Reinforcement Learning

IMI supports multiple RL approaches:

1. **DPO (Direct Preference Optimization)**
   - No reward model needed
   - Stable training
   - Good for preference alignment

2. **ORPO (Odds Ratio Preference Optimization)**
   - Combines SFT and preference learning
   - Single training stage
   - More memory efficient

3. **PPO (Proximal Policy Optimization)** *(coming soon)*
   - Classic RLHF approach
   - Requires reward model
   - Most flexible

## 📈 Experiment Tracking

```bash
# Enable Weights & Biases logging
python scripts/training/run_training.py --all --wandb

# View training metrics
wandb login
# Then check your W&B dashboard
```

## 🚀 Deployment

### Local Inference
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = PeftModel.from_pretrained(model, "outputs/imi-medical/sft")

# Generate
messages = [{"role": "user", "content": "What are the symptoms of diabetes?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False)
outputs = model.generate(tokenizer(prompt, return_tensors="pt").input_ids)
print(tokenizer.decode(outputs[0]))
```

### vLLM Serving (Production)
```bash
python -m vllm.entrypoints.openai.api_server \
    --model outputs/imi-medical/sft \
    --port 8000
```

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Built for better medical AI 🏥**
