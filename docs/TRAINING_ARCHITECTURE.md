# IMI Training Architecture — Mistral 7B Full Fine-Tuning

## Training Pipeline Overview

```
+===========================================================================+
|               IMI TRAINING PIPELINE — Mistral 7B Full Fine-Tuning         |
+===========================================================================+

   PHASE 1: DATA COLLECTION          PHASE 2: PREPROCESSING
   (Open/Verified Sources)            (Quality Assurance)
   ========================          =======================

   +-------------------+             +-------------------------+
   | HuggingFace Hub   |             |   PREPROCESSING PIPELINE|
   | (No login needed) |             |                         |
   | - MedQA           |             |  1. Structure Validation|
   | - MedMCQA         |------------>|     - Required fields   |
   | - PubMedQA        |             |     - Type checking     |
   | - HealthCareMagic |             |     - Length bounds      |
   | - WikiDoc         |             |                         |
   | - DrugBank        |             |  2. Text Cleaning       |
   | - 30+ datasets    |             |     - Strip HTML/markup |
   +-------------------+             |     - Fix encoding      |
                                     |     - Normalize unicode |
   +-------------------+             |     - Collapse spaces   |
   | GitHub Public     |             |                         |
   | - MTSamples       |             |  3. Garbage Detection   |
   | - ChatDoctor      |------------>|     - Empty/null check  |
   | - MedCalc-Bench   |             |     - Pattern matching  |
   | - Synthea         |             |     - Encoding corrupt. |
   | - UCSD Dialogs    |             |     - Placeholder text  |
   +-------------------+             |                         |
                                     |  4. Quality Scoring     |
   +-------------------+             |     - Instruction len   |
   | US Gov Open Data  |             |     - Output coherence  |
   | - CDC (no key)    |------------>|     - Sentence structure|
   | - CMS (no key)    |             |     - Content diversity |
   | - OpenFDA (no key)|             |     - Score threshold   |
   | - NLM RxNorm      |             |                         |
   +-------------------+             |  5. Deduplication       |
                                     |     - SHA-256 hash      |
   +-------------------+             |     - instruction+input |
   | Synthetic Data    |             |                         |
   | - Template-based  |------------>|  6. Train/Val Split     |
   | - Triage cases    |             |     - 90/10 ratio       |
   | - Drug interacts  |             |     - Shuffled          |
   | - USMLE Q&A       |             +-------------------------+
   +-------------------+                       |
                                               v
                                    +---------------------+
                                    |   data/final/       |
                                    |   *_train.json      |
                                    |   *_val.json        |
                                    | (Clean, validated,  |
                                    |  deduplicated)      |
                                    +---------------------+
                                               |
                                               v
   =================================================================
   PHASE 3: FOUNDATION TRAINING (Full Fine-Tuning — 8× A100 80GB)
   =================================================================

   +---------------------------------------------------------------+
   |                                                               |
   |   Base Model: Mistral-7B-Instruct-v0.3                       |
   |   (7B parameters — ALL trainable, no LoRA)                   |
   |                                                               |
   |   +-------------------+    +-----------------------------+    |
   |   | Model Loading     |    | Multi-GPU Optimizations     |    |
   |   |                   |    |                             |    |
   |   | - BFloat16        |    | - Flash Attention 2         |    |
   |   | - No quantization |    | - TF32 matmul enabled       |    |
   |   | - Full parameters |    | - Gradient checkpointing    |    |
   |   | - DeepSpeed FSDP  |    | - Sequence packing          |    |
   |   +-------------------+    | - Pin memory + 8 workers    |    |
   |                            | - DeepSpeed ZeRO-3          |    |
   |                            +-----------------------------+    |
   |                                                               |
   |   Training Config (8× A100 80GB — PRIMARY):                  |
   |   +-------------------------------------------------------+  |
   |   | Epochs:            3                                   |  |
   |   | Per-device batch:  8                                   |  |
   |   | Grad accumulation: 2  (effective batch = 128)          |  |
   |   | Learning rate:     2e-5 (cosine schedule)              |  |
   |   | Warmup:            3%                                  |  |
   |   | Max seq length:    4096                                |  |
   |   | Precision:         BFloat16                            |  |
   |   | Strategy:          DeepSpeed ZeRO-3                    |  |
   |   | Per GPU memory:    ~30 GB  (50 GB headroom on 80GB)    |  |
   |   | Launch:            torchrun --nproc_per_node=8         |  |
   |   +-------------------------------------------------------+  |
   |                                                               |
   |   ZeRO-3 memory breakdown (per GPU, 7B model):               |
   |   +-------------------------------------------------------+  |
   |   | Model shard (BF16):   7B × 2B / 8 GPUs  ≈  1.75 GB  |  |
   |   | Gradient shard:                          ≈  1.75 GB  |  |
   |   | Optimizer shard (Adam 32-bit / 8):       ≈  7.00 GB  |  |
   |   | Activations (batch=8, seq=4096, gc):     ≈ 18–22 GB  |  |
   |   | NCCL buffers:                            ≈  1 GB     |  |
   |   | Total per GPU:                           ≈ 30–33 GB  ✓ |  |
   |   +-------------------------------------------------------+  |
   |                                                               |
   |   Speed estimates (8× A100 80GB):                            |
   |   +-------------------------------------------------------+  |
   |   | 500K examples, 3 epochs:  ~20 min,  cost ~$1          |  |
   |   | 4M  examples, 3 epochs:  ~2.5 hrs,  cost ~$24         |  |
   |   | Full ~5M corpus, 3 epochs: ~3 hrs,   cost ~$30        |  |
   |   +-------------------------------------------------------+  |
   |                                                               |
   +---------------------------------------------------------------+
                          |
                          v
               +---------------------+
               | models/foundation/  |
               | (Full fine-tuned    |
               |  Mistral 7B)        |
               +---------------------+
                          |
                          v
   =================================================================
   PHASE 4: ORPO SAFETY ALIGNMENT (Full Fine-Tuning — 1× A100 40GB)
   =================================================================

   +---------------------------------------------------------------+
   |                                                               |
   |  ORPO — Odds Ratio Preference Optimization                    |
   |  (replaces DPO — combined SFT + preference loss)             |
   |                                                               |
   |  Input: Safety pairs (prompt, chosen=safe, rejected=unsafe)   |
   |                                                               |
   |  +------------------------------------------+                |
   |  | Single Training Model (no ref model)     |                |
   |  | (all params trainable)                   |                |
   |  |                                          |                |
   |  | Joint loss = SFT loss + odds-ratio loss  |                |
   |  | No reference model → saves ~14 GB GPU    |                |
   |  +------------------------------------------+                |
   |                                                               |
   |  Safety Categories (30+ seed pairs, expand to 500+):         |
   |  +-------------------------------------------------------+   |
   |  | - Medication dosing (hedge vs specific dose)            |   |
   |  | - Emergency symptoms (escalate vs dismiss)              |   |
   |  | - Suicide/self-harm (crisis resources vs ignore)        |   |
   |  | - Diagnosis requests (caveat vs confident dx)           |   |
   |  | - Off-label drug use (caveat vs recommend)              |   |
   |  | - Scope boundaries (decline vs attempt)                 |   |
   |  | - Overconfident language (calibrated vs absolute)       |   |
   |  +-------------------------------------------------------+   |
   |                                                               |
   |  Config: beta=0.1, lr=8e-6, BFloat16, Flash Attn 2           |
   |  Hardware: 1× A100 40GB (model=14GB + optimizer fits easily)  |
   |                                                               |
   +---------------------------------------------------------------+
                          |
                          v
               +----------------------+
               | models/orpo_aligned/ |
               | (Safety-aligned      |
               |  Mistral 7B)         |
               +----------------------+
                          |
                          v
   =================================================================
   PHASE 5: EVALUATION
   =================================================================

   +---------------------------------------------------------------+
   |                                                               |
   |  EVALUATION METRICS (Pass/Fail Gates)                         |
   |                                                               |
   |  +----------------------------+  +------------------------+  |
   |  | Clinical Quality           |  | Safety Metrics          |  |
   |  |                            |  |                        |  |
   |  | Perplexity     <= 8.0      |  | Unsafe claim   <= 5%  |  |
   |  | MCQ/USMLE acc  >= 55%      |  | Emergency miss <= 2%  |  |
   |  | Triage F1      >= 70%      |  | Disclaimer     >= 80% |  |
   |  | ROUGE-L F1     >= 0.20     |  | Crisis recall  = 100% |  |
   |  +----------------------------+  | Emerg. recall  >= 95%  |  |
   |                                  +------------------------+  |
   |                                                               |
   |  ALL metrics must PASS before deployment                      |
   |                                                               |
   +---------------------------------------------------------------+
                          |
                    PASS? |
                   +------+------+
                   |             |
                  YES           NO
                   |             |
                   v             v
           +-----------+  +------------------+
           | DEPLOY    |  | Investigate &    |
           | to prod   |  | retrain          |
           +-----------+  +------------------+


   =================================================================
   HARDWARE REQUIREMENTS
   =================================================================

   +---------------------------------------------------------------+
   |                                                               |
   |  FOUNDATION:  8× A100 80GB  (primary — your setup)            |
   |  ORPO/ADAPTERS: 1× A100 40GB or 80GB (any single GPU)         |
   |                                                               |
   |  8× A100 80GB — Foundation full fine-tuning:                  |
   |  - DeepSpeed ZeRO-3, no CPU offload needed                    |
   |  - Per GPU: ~30–33 GB VRAM (50 GB headroom)                   |
   |  - Launch: torchrun --nproc_per_node=8 train_foundation.py    |
   |  - 500K examples, 3 epochs:  ~20 min,  cost ~$1              |
   |  - 4M  examples, 3 epochs:  ~2.5 hrs,  cost ~$24             |
   |  - Full ~5M corpus, 3 epochs: ~3 hrs,   cost ~$30            |
   |                                                               |
   |  1× A100 40GB — ORPO safety alignment:                        |
   |  - Full 7B param training (BF16, no quantization)             |
   |  - ~38 GB VRAM (optimizer sharded via ZeRO-2)                 |
   |  - 30 pairs → ~500+ pairs: ~30 min, cost ~$1                 |
   |                                                               |
   |  1× A100 40GB — QLoRA domain adapters (6 adapters):           |
   |  - Mistral 7B in 4-bit NF4 ≈ 6 GB base                       |
   |  - batch=4, seq=2048: ~28 GB total VRAM                       |
   |  - All 6 adapters: ~2 hrs total, cost ~$4                    |
   |                                                               |
   |  Key GPU Features Utilized:                                   |
   |  - BFloat16 native (no quantization needed for full FT)       |
   |  - TF32 matmul (1.5× speedup over FP32 on A100)              |
   |  - Flash Attention 2 (memory-efficient, long sequences)       |
   |  - DeepSpeed ZeRO-3 (foundation) / ZeRO-2 (ORPO/adapters)    |
   |  - NVLink / NVSwitch for fast inter-GPU AllReduce             |
   |                                                               |
   +---------------------------------------------------------------+


   =================================================================
   DATA FLOW SUMMARY
   =================================================================

   Open Data Sources (HF, GitHub, CDC, CMS, OpenFDA)
          |
          v
   Preprocessing (validate, clean, filter, deduplicate)
          |
          v
   data/final/ (clean training data)
          |
          v
   Foundation Training (full FT, 8× A100 80GB, ZeRO-3, all 7B params)
          |
          v
   ORPO Safety Alignment (full FT, A100 40GB, safety pairs)
          |
          v
   Evaluation (perplexity, MCQ, triage, safety audit)
          |
          v
   Production Deployment (if all metrics pass)
```

## Comparison: QLoRA Only vs Full Fine-Tuning (Mistral 7B)

| Aspect | QLoRA Only (Path A) | Full FT + ORPO + QLoRA (Path B) |
|--------|---------------------|----------------------------------|
| Parameters trained | ~0.5% (LoRA adapters) | 100% (all 7B) foundation + adapters |
| Foundation hardware | None needed | **8× A100 80GB** (your setup), ZeRO-3 |
| Adapter hardware | 1× A100 40GB | 1× A100 40GB |
| Precision | 4-bit NF4 | BFloat16 (foundation), 4-bit (adapters) |
| Foundation time | — | ~20 min (500K) / ~3 hrs (5M examples) |
| Foundation cost | $0 | ~$1–30 depending on dataset size |
| Adapter cost | ~$4 total | ~$4 total |
| Quality ceiling | Good for POC | Higher — full medical domain shift |
| Safety alignment | ORPO (1× A100 40GB) | ORPO (1× A100 40GB) |
| Deployment | 6 adapters hot-swap | 6 adapters hot-swap |
| Recommended for | Fast iteration | **Production** (your path) |
