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
   PHASE 3: FOUNDATION TRAINING (Full Fine-Tuning — 1× A100 80GB)
   =================================================================

   +---------------------------------------------------------------+
   |                                                               |
   |   Base Model: Mistral-7B-Instruct-v0.3                       |
   |   (7B parameters — ALL trainable, no LoRA)                   |
   |                                                               |
   |   +-------------------+    +-----------------------------+    |
   |   | Model Loading     |    | A100/H100 Optimizations     |    |
   |   |                   |    |                             |    |
   |   | - BFloat16        |    | - Flash Attention 2         |    |
   |   | - No quantization |    | - TF32 matmul enabled       |    |
   |   | - Full parameters |    | - Gradient checkpointing    |    |
   |   | - device_map=auto |    | - Sequence packing          |    |
   |   +-------------------+    | - Pin memory + 4 workers    |    |
   |                            | - DeepSpeed CPU offload     |    |
   |                            +-----------------------------+    |
   |                                                               |
   |   Training Config (single A100 80GB):                        |
   |   +-------------------------------------------------------+  |
   |   | Epochs:           3                                    |  |
   |   | Per-device batch: 4                                    |  |
   |   | Grad accumulation: 4  (effective batch = 16)           |  |
   |   | Learning rate:    2e-5 (cosine schedule)               |  |
   |   | Warmup:           3%                                   |  |
   |   | Max seq length:   2048                                 |  |
   |   | Precision:        BFloat16                             |  |
   |   | Optimizer:        AdamW (CPU offloaded via ZeRO-3)     |  |
   |   | GPU memory:       ~38 GB  (42 GB headroom on 80GB)     |  |
   |   +-------------------------------------------------------+  |
   |                                                               |
   |   Optional multi-GPU (faster):                               |
   |   +-------------------------------------------------------+  |
   |   | torchrun --nproc_per_node=4 train_foundation.py        |  |
   |   | + configs/deepspeed_zero3.json                         |  |
   |   | 4× A100 80GB: ~2× faster, each GPU uses ~20 GB        |  |
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
   |  MINIMUM: 1× A100 40GB (QLoRA adapter training)               |
   |  RECOMMENDED: 1× A100 80GB (foundation + adapters)            |
   |                                                               |
   |  Single A100 80GB (foundation training):                      |
   |  - Full 7B parameter training with DeepSpeed CPU offload      |
   |  - ~38 GB GPU VRAM, ~56 GB CPU RAM for optimizer              |
   |  - Estimated: 2 hrs (500K examples) / 8 hrs (4M examples)     |
   |                                                               |
   |  Single A100 40GB (adapter QLoRA):                            |
   |  - Mistral 7B in 4-bit NF4 ≈ 6 GB base                       |
   |  - batch=4, seq=2048: ~28 GB total VRAM                       |
   |  - Estimated: 1–3 hrs per adapter                             |
   |                                                               |
   |  8× A100 80GB (your setup — fast foundation):                 |
   |  - ZeRO-3: each GPU holds 7B/8 = ~1.75 GB model shard        |
   |  - Foundation 4M examples: ~1.5 hrs, cost ~$24               |
   |                                                               |
   |  Key GPU Features Utilized:                                   |
   |  - BFloat16 native (no quantization needed for FT)            |
   |  - TF32 matmul (1.5x speedup over FP32 on A100/H100)         |
   |  - Flash Attention 2 (memory-efficient attention)             |
   |  - DeepSpeed ZeRO-3 + CPU optimizer offload                   |
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
   Foundation Training (full FT, A100 80GB, BF16, all 7B params)
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
| Foundation GPU | None needed | 1× A100 80GB + CPU offload |
| Adapter GPU | 1× A100 40GB | 1× A100 40GB |
| Precision | 4-bit NF4 | BFloat16 (foundation), 4-bit (adapters) |
| Foundation cost | $0 | ~$4–24 |
| Adapter cost | ~$4 total | ~$4 total |
| Quality ceiling | Good for MVP | Higher — full medical domain shift |
| Safety alignment | ORPO (A100 40GB) | ORPO (A100 40GB) |
| Deployment | 6 adapters hot-swap | 6 adapters hot-swap |
| Recommended for | POC, fast iteration | Production, clinical grade |
