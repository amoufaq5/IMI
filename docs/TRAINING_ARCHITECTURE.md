# IMI Training Architecture — Full Fine-Tuning on H100

## Training Pipeline Overview

```
+===========================================================================+
|                    IMI TRAINING PIPELINE (H100 Full FT)                    |
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
   PHASE 3: FOUNDATION TRAINING (Full Fine-Tuning on H100)
   =================================================================

   +---------------------------------------------------------------+
   |                                                               |
   |   Base Model: Mistral 7B Instruct v0.3                     |
   |   (7.3B parameters — ALL trainable, no LoRA)                 |
   |                                                               |
   |   +-------------------+    +-----------------------------+    |
   |   | Model Loading     |    | H100 Optimizations          |    |
   |   |                   |    |                             |    |
   |   | - BFloat16        |    | - Flash Attention 2         |    |
   |   | - No quantization |    | - TF32 matmul enabled       |    |
   |   | - Full parameters |    | - torch.compile()           |    |
   |   | - device_map=auto |    | - Fused AdamW optimizer     |    |
   |   +-------------------+    | - Gradient checkpointing    |    |
   |                            | - Sequence packing          |    |
   |                            | - Pin memory + 8 workers    |    |
   |                            +-----------------------------+    |
   |                                                               |
   |   Training Config:                                            |
   |   +-------------------------------------------------------+  |
   |   | Epochs:           2                                    |  |
   |   | Per-device batch: 4                                    |  |
   |   | Grad accumulation: 8  (effective batch = 32 * N_GPUs)  |  |
   |   | Learning rate:    2e-5 (cosine schedule)               |  |
   |   | Warmup:           5%                                   |  |
   |   | Max seq length:   4096                                 |  |
   |   | Precision:        BFloat16                             |  |
   |   | Optimizer:        AdamW (fused, torch)                 |  |
   |   | Weight decay:     0.01                                 |  |
   |   +-------------------------------------------------------+  |
   |                                                               |
   |   Multi-GPU Strategy (for 2+ H100s):                         |
   |   +-------------------------------------------------------+  |
   |   | DeepSpeed ZeRO Stage 3 or PyTorch FSDP                |  |
   |   | - Shards optimizer state across GPUs                   |  |
   |   | - Shards gradients across GPUs                         |  |
   |   | - Shards parameters across GPUs                        |  |
   |   | - Enables full FT of 46.7B model on H100 cluster       |  |
   |   +-------------------------------------------------------+  |
   |                                                               |
   +---------------------------------------------------------------+
                          |
                          v
               +---------------------+
               | models/foundation/  |
               | (Full fine-tuned    |
               |  Mistral 7B)      |
               +---------------------+
                          |
                          v
   =================================================================
   PHASE 4: DPO SAFETY ALIGNMENT (Full Fine-Tuning on H100)
   =================================================================

   +---------------------------------------------------------------+
   |                                                               |
   |  Direct Preference Optimization (DPO)                         |
   |                                                               |
   |  Input: Safety pairs (prompt, chosen=safe, rejected=unsafe)   |
   |                                                               |
   |  +-------------------------+  +----------------------------+  |
   |  | Training Model          |  | Reference Model (frozen)   |  |
   |  | (all params trainable)  |  | (same foundation weights)  |  |
   |  |                         |  |                            |  |
   |  | Learns to prefer safe   |  | Provides baseline policy   |  |
   |  | responses over unsafe   |  | for KL divergence penalty  |  |
   |  +-------------------------+  +----------------------------+  |
   |                                                               |
   |  Safety Categories:                                           |
   |  +-------------------------------------------------------+   |
   |  | - Medication dosing (hedge vs specific dose)            |   |
   |  | - Emergency symptoms (escalate vs dismiss)              |   |
   |  | - Suicide/self-harm (crisis resources vs ignore)        |   |
   |  | - Diagnosis requests (caveat vs confident dx)           |   |
   |  | - Off-label drug use (caveat vs recommend)              |   |
   |  | - Scope boundaries (decline vs attempt)                 |   |
   |  | - Overconfident language (calibrated vs absolute)        |   |
   |  +-------------------------------------------------------+   |
   |                                                               |
   |  Config: beta=0.1, lr=5e-7, BFloat16, Flash Attn 2           |
   |                                                               |
   +---------------------------------------------------------------+
                          |
                          v
               +---------------------+
               | models/dpo_aligned/ |
               | (Safety-aligned     |
               |  Mistral 7B)      |
               +---------------------+
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
   |  RECOMMENDED: NVIDIA H100 80GB HBM3                           |
   |                                                               |
   |  Single H100:                                                 |
   |  - Foundation training with gradient checkpointing            |
   |  - ~80GB VRAM utilization                                     |
   |  - Estimated: 24-48 hours for 2 epochs on ~3M examples        |
   |                                                               |
   |  Multi-H100 (recommended for production):                     |
   |  - 4x H100 with DeepSpeed ZeRO-3                             |
   |  - ~20GB VRAM per GPU (params sharded)                        |
   |  - Estimated: 6-12 hours for 2 epochs on ~3M examples         |
   |  - 8x H100: ~3-6 hours                                       |
   |                                                               |
   |  Key H100 Features Utilized:                                  |
   |  - BFloat16 native (no quantization needed)                   |
   |  - TF32 matmul (1.5x speedup over FP32)                      |
   |  - Flash Attention 2 (memory-efficient attention)             |
   |  - 3.35 TB/s memory bandwidth                                 |
   |  - torch.compile() graph optimization                         |
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
   Foundation Training (full FT, H100, BF16, all params)
          |
          v
   DPO Safety Alignment (full FT, H100, safety pairs)
          |
          v
   Evaluation (perplexity, MCQ, triage, safety audit)
          |
          v
   Production Deployment (if all metrics pass)
```

## Comparison: Previous (LoRA) vs Current (Full FT on H100)

| Aspect | Previous (LoRA/QLoRA) | Current (Full FT on H100) |
|--------|----------------------|---------------------------|
| Parameters trained | ~0.5% (LoRA adapters) | 100% (all 46.7B) |
| Precision | 4-bit NF4 quantized | BFloat16 native |
| GPU | A100 80GB | H100 80GB |
| Memory | ~52GB (quantized) | ~80GB (full precision) |
| Quality ceiling | Limited by frozen base | Maximized — full capacity |
| Training speed | Fast per step | Leverages H100 hardware |
| Adapter complexity | 6 separate adapters | Single unified model |
| Deployment | Adapter hot-swap | Single model, simpler |
| Safety alignment | LoRA on top of LoRA | Full parameter DPO |
