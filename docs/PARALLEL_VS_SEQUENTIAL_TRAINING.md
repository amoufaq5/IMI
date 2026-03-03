# Parallel vs Sequential Training: Analysis for IMI LoRA Adapters

## Overview

IMI trains **6 domain-specific LoRA adapters** on top of **Meditron-7B**. This document compares two strategies for training them: **sequential (one-after-another)** and **parallel (simultaneous)**, analysing efficiency, hardware requirements, and quality trade-offs.

> **Current default**: Parallel multi-GPU training (`--parallel` flag). Each adapter trains on a separate GPU via subprocess isolation.

---

## 1. Sequential Training (Fallback)

### How It Works

Each adapter is trained one at a time on a single GPU. The base model is loaded once, LoRA weights are attached, trained, saved, then reset before the next adapter.

```
┌──────────┐   ┌──────────┐   ┌──────────┐       ┌──────────┐
│ Adapter 1 │──▶│ Adapter 2 │──▶│ Adapter 3 │──...──▶│ Adapter 6 │
│ (triage)  │   │ (pharma)  │   │ (clinical)│       │ (research)│
└──────────┘   └──────────┘   └──────────┘       └──────────┘
     GPU 0          GPU 0          GPU 0               GPU 0
```

### Time Estimate (Meditron-7B, 4-bit QLoRA, single A100-40GB)

| Adapter | Dataset Size | Epochs | Est. Time |
|---------|-------------|--------|----------|
| patient_triage | ~50K examples | 3 | ~2.5 hrs |
| clinical_pharmacist | ~30K examples | 3 | ~1.5 hrs |
| clinical_decision | ~40K examples | 4 | ~2.5 hrs |
| education | ~35K examples | 3 | ~1.8 hrs |
| regulatory_qa | ~15K examples | 3 | ~1.0 hrs |
| research | ~20K examples | 4 | ~1.5 hrs |
| **Total** | | | **~10.8 hrs** |

Add ~5 min per adapter for model reset = **~11 hrs total wall time**.

### Pros

- **Simplest setup** — single GPU, no distributed infrastructure
- **No interference** — each adapter trains independently, no gradient cross-talk
- **Easy debugging** — one training run at a time, clear logs
- **Lower peak memory** — only one LoRA module in memory at a time (~8 MB per adapter at r=16)
- **Reproducible** — deterministic ordering, easy to resume from any adapter

### Cons

- **Wall-clock time is additive** — total time = sum of all adapter training times
- **GPU utilization drops** between adapters (save/load overhead)
- **Base model reload** was happening for each adapter (now fixed — we reset only LoRA weights)

---

## 2. Parallel Training

### Option A: Multi-GPU Parallel (One Adapter Per GPU) — ✅ IMPLEMENTED

Each adapter trains on a separate GPU as an independent subprocess. The base model is replicated per-GPU via `CUDA_VISIBLE_DEVICES` isolation.

```
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Adapter 1 │  │ Adapter 2 │  │ Adapter 3 │  ...
│   GPU 0   │  │   GPU 1   │  │   GPU 2   │
└──────────┘  └──────────┘  └──────────┘
     ▼              ▼              ▼
  Save to        Save to        Save to
  /adapters/     /adapters/     /adapters/
```

**Usage:**
```bash
python scripts/training/train_lora.py --adapter all --parallel
```

**Time estimate**: ~2.5 hrs (limited by the slowest adapter: clinical_decision at 4 epochs) — a **4× speedup**.

**Hardware required**: 6× A100-40GB (or 3× A100-80GB). 7B model in 4-bit ≈ 4 GB VRAM per GPU.

### Option B: Single-GPU Time-Sliced Parallel

Adapters share a single GPU via gradient accumulation interleaving. The base model stays loaded; LoRA weights for all adapters are in memory but only one computes gradients per micro-step.

```
Step 1: adapter_1 forward/backward (LoRA weights 1)
Step 2: adapter_2 forward/backward (LoRA weights 2)
Step 3: adapter_3 forward/backward (LoRA weights 3)
...
Every N steps: optimizer.step() for all adapters
```

**Time estimate**: ~12 hrs (slightly slower than sequential due to switching overhead).

**Memory**: Base model (4-bit) ≈ 4 GB + 6 × LoRA weights ≈ 48 MB + optimizer states ≈ 300 MB. **Fits on a single A100-40GB** but is memory-tight.

### Option C: PEFT Multi-Adapter Training (Recommended for IMI)

The `peft` library supports **multiple LoRA adapters** on a single model. All adapters share the base model weights but maintain separate LoRA parameters. You can switch between them with `model.set_adapter("adapter_name")`.

```python
from peft import PeftModel, LoraConfig

# Load base model once
model = load_base_model()

# Add all adapters
for adapter_name, config in ADAPTER_CONFIGS.items():
    model.add_adapter(adapter_name, LoraConfig(**config))

# Training loop
for epoch in range(num_epochs):
    for adapter_name in adapters:
        model.set_adapter(adapter_name)
        for batch in dataloaders[adapter_name]:
            loss = model(batch)
            loss.backward()
            optimizers[adapter_name].step()
```

**Time estimate**: ~7-8 hrs (overlap from keeping model hot, no reloads).

**Memory**: Single 7B base model (~4 GB in 4-bit) + 6 adapter weight sets (~48 MB). Fits on a single A100-40GB.

---

## 3. Head-to-Head Comparison

| Dimension | Sequential | Multi-GPU Parallel ✅ | Time-Sliced | PEFT Multi-Adapter |
|-----------|-----------|-------------------|-------------|-------------------|
| **Wall time** | ~11 hrs | **~2.5 hrs** | ~12 hrs | ~7-8 hrs |
| **Speedup** | 1× | **4×** | 0.9× | **1.5×** |
| **GPUs needed** | 1 | 6 | 1 | 1 |
| **Peak VRAM** | ~6 GB | ~6 GB × N | ~8 GB | ~7 GB |
| **Cloud cost (A100-40GB)** | ~$17 | **~$15** | ~$18 | ~$12 |
| **Adapter interference** | None | None | Possible | None (separate optimizers) |
| **Implementation complexity** | Low | **Low (implemented)** | High | Medium |
| **Resume capability** | Easy | Per-GPU | Complex | Moderate |
| **Quality risk** | None | None | Gradient noise | None |

> **Cloud cost** estimated at ~$1.50/hr per A100-40GB (on-demand). Multi-GPU parallel is both **faster and cheaper** than sequential.

---

## 4. Quality Considerations

### Does Parallel Training Affect Model Quality?

**Multi-GPU parallel (Option A)**: **No quality difference**. Each adapter trains independently, just on different hardware. Mathematically identical to sequential.

**Time-sliced parallel (Option B)**: **Potential quality degradation**. Interleaving gradients from different tasks can cause:
- Gradient interference between adapters
- Suboptimal convergence if learning rates aren't tuned per-adapter
- Memory contention leading to smaller effective batch sizes

**PEFT multi-adapter (Option C)**: **No quality difference** if each adapter has its own optimizer and the adapters are switched (not merged) between batches. The base model weights are frozen and shared read-only.

### Medical Safety Implication

For a medical LLM, training quality directly impacts patient safety. We recommend:
1. **Always evaluate each adapter independently** after training (using the upgraded evaluation script with USMLE accuracy, triage F1, safety audit, and ROUGE)
2. **Compare parallel-trained adapters against sequential baselines** before deployment
3. **Never deploy an adapter that scores below threshold** on safety metrics regardless of training method

---

## 5. Recommendation for IMI

### Current Setup — Parallel Multi-GPU (Implemented)

**Multi-GPU parallel training is the default** via `--parallel` flag.

```bash
# Launch all 6 adapters in parallel across available GPUs
python scripts/training/train_lora.py --adapter all --parallel

# Or train a single adapter on a specific GPU
python scripts/training/train_lora.py --adapter patient_triage --gpu 2
```

- Each adapter runs as an isolated subprocess with `CUDA_VISIBLE_DEVICES`
- The parent process monitors all children and reports success/failure
- If fewer GPUs than adapters, adapters are round-robin assigned (2 per GPU)
- **Recommended**: 4–6× A100-40GB for 7B model with 4-bit QLoRA
- Wall time: **~2.5 hrs** (vs ~11 hrs sequential) — **4× speedup**

### Fallback — Sequential (Single GPU)

Used automatically when `--parallel` is not set or only 1 GPU is available.

- Base model loaded once, LoRA weights reset between adapters
- Suitable for development/debugging on a single A100-40GB or RTX 3090

### Future — PEFT Multi-Adapter (Continuous Training)

For incremental production updates:

- Keep all adapters in memory on a single node
- Fine-tune on new data as it arrives
- Evaluate after each update cycle
- Roll back any adapter that degrades on safety metrics

---

## 6. Summary

| Question | Answer |
|----------|--------|
| Is parallel training faster? | **Yes** — 4× with multi-GPU (implemented) |
| Does it affect quality? | **No** — each adapter trains independently per-GPU |
| What does IMI use? | **Multi-GPU parallel** (`--parallel`) as default |
| Base model? | **Meditron-7B** with 4-bit QLoRA (r=16, alpha=32) |
| Hardware needed? | 1× A100-40GB minimum, 6× for parallel |
| Is time-sliced worth it? | **No** — complexity + quality risk, minimal speed gain |
