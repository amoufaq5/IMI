# IMI — Additional Recommendations & Next Steps

## Already Implemented in This Upgrade

| Feature | Status | Location |
|---------|--------|----------|
| Mistral 7B base model (MVP/POC) | ✅ Done | `settings.py`, `meditron.py`, all training scripts |
| 40+ dataset catalogue | ✅ Done | `collect_datasets.py` |
| 3-stage training pipeline | ✅ Done | `train_foundation.py` → `train_dpo.py` → `train_lora.py` |
| DPO safety alignment | ✅ Done | `train_dpo.py` (30 seed pairs, expandable) |
| vLLM inference with LoRA hot-swap | ✅ Done | `start_vllm.sh` |
| Wave-based parallel training | ✅ Done | `train_lora.py` (1 adapter per GPU per wave) |
| Input/Output guardrails | ✅ Done | `guardrails.py` (crisis, emergency, scope, PHI, overconfidence) |
| Evaluation thresholds | ✅ Done | `evaluate_adapter.py` (9 pass/fail gates) |
| Weights & Biases integration | ✅ Done | All training scripts report to W&B |
| Confidence softening | ✅ Done | `OutputGuardrail` rewrites overconfident language |
| Investor demo materials | ✅ Done | `INVESTOR_DEMO.md` |

---

## Further Recommendations

### 1. Expand ORPO Safety Pairs to 500+
**Priority: HIGH** | **Effort: 2-3 days**

The current 30 seed pairs in `train_dpo.py` (ORPO mode) cover the core safety categories but
production training needs 500+ pairs for robust alignment. Recommended approach:
- Have a medical professional review and expand the seed pairs
- Add pairs for: pediatric dosing, geriatric concerns, pregnancy safety, drug withdrawal
- Use red-teaming to discover edge cases the model handles poorly
- Export pairs: `python scripts/training/train_dpo.py export`

### 2. Add Automated Red-Teaming Pipeline
**Priority: HIGH** | **Effort: 1 week**

Create an automated adversarial testing script that:
- Generates jailbreak attempts (prompt injection, role-play attacks)
- Tests boundary conditions for each guardrail pattern
- Measures false positive/negative rates
- Runs as CI/CD gate before deployment

### 3. Implement RAG-Augmented Generation for Drug Database
**Priority: HIGH** | **Effort: 3-5 days**

The current knowledge graph is great for structured queries, but for real-time drug
information, implement a RAG pipeline over:
- DrugBank database (already in dataset catalogue)
- FDA drug labels (DailyMed API)
- UpToDate/Lexicomp summaries (if licensed)

This ensures the LLM always has current drug information, not just training-time knowledge.

### 4. Add Continuous Learning Pipeline
**Priority: MEDIUM** | **Effort: 1-2 weeks**

After deployment, implement a feedback loop:
- Log user thumbs-up/down on responses
- Collect correction data from physician reviews
- Periodic adapter re-training on new data (monthly or quarterly)
- A/B test new adapters against current production adapters

### 5. Multi-Language Support
**Priority: MEDIUM** | **Effort: 1-2 weeks**

Mistral 7B has strong multilingual capability (trained on multilingual data). To enable:
- Add Arabic and French medical datasets to the training pipeline
- Create language-detection guardrails
- Test safety patterns in target languages
- Update prompt templates for multilingual system prompts

### 6. Structured Output (JSON Mode)
**Priority: MEDIUM** | **Effort: 2-3 days**

For integration with clinical systems, add structured output support:
- Triage results as JSON (urgency, recommendations, red_flags)
- Drug interaction checks as structured data
- ICD-10 code suggestions with confidence scores
- vLLM supports guided generation / JSON schema enforcement

### 7. Model Merging (DPO + Foundation)
**Priority: LOW** | **Effort: 1 day**

Instead of keeping DPO as a separate LoRA layer, consider merging the DPO adapter
into the foundation weights using `model.merge_and_unload()`. This:
- Reduces inference overhead (one less adapter to load)
- Simplifies the deployment architecture
- Can be done with PEFT's built-in merge functionality

### 8. Quantization Optimization for Inference
**Priority: LOW** | **Effort: 1-2 days**

For cost-efficient inference:
- Test AWQ quantization (faster than GPTQ on vLLM)
- Benchmark GGUF format for edge deployment
- Consider speculative decoding for latency reduction
- Test FP8 quantization on H100 GPUs

### 9. Add Patient Consent & Data Governance Layer
**Priority: MEDIUM** | **Effort: 1 week**

For enterprise/hospital deployment:
- Implement explicit patient consent tracking before storing any data
- Add data retention policies (auto-delete after N days)
- Create data export functionality (GDPR right to erasure)
- Add role-based access control to patient data

### 10. Monitoring & Alerting Dashboard
**Priority: MEDIUM** | **Effort: 3-5 days**

Set up production monitoring:
- Grafana dashboard for inference latency, throughput, error rates
- Alert on safety metric degradation (rising unsafe claim rate)
- Track adapter usage patterns across user types
- Monitor GPU utilization and auto-scaling triggers

---

## Budget Estimate for Full Implementation

| Phase | Cost | Timeline |
|-------|------|----------|
| Foundation full FT (8× A100 80GB, 5M corpus) | ~$30 | ~3 hrs |
| ORPO alignment + 6 LoRA adapters (A100 40GB) | ~$5 | ~3 hrs |
| Full training pipeline | ~$35 | 1 day |
| Red-teaming + safety expansion | ~$50 | 3 days |
| Production inference (monthly, 1× A100 40GB) | ~$500–800/mo | Ongoing |
| Continuous learning pipeline | ~$50/quarter | Quarterly |
| **Total Year 1** | **~$7,000–10,000** | |

---

## Architecture Comparison: Before vs Now

| Aspect | Before (Meditron-7B) | Now (Mistral 7B — MVP) |
|--------|---------------------|------------------------|
| Base model | Meditron-7B (7B params) | Mistral-7B-Instruct-v0.3 (7B params) |
| License | LLaMA license (restricted) | Apache 2.0 (commercial) |
| Architecture | Dense transformer | Dense transformer (SwiGLU FFN) |
| Context window | 4,096 tokens | 32,768 tokens |
| LoRA rank | r=16, α=32 | r=32–64, α=64–128 |
| LoRA targets | q_proj, v_proj | q,k,v,o_proj + gate/up/down_proj |
| Training pipeline | Single-stage LoRA | 3-stage: Foundation → ORPO → LoRA |
| Safety alignment | None (rule engine only) | ORPO + guardrails + rule engine + verifier |
| Alignment method | — | ORPO (no reference model → saves ~14GB GPU) |
| Inference | Direct HF transformers | vLLM with LoRA hot-swapping |
| Datasets | ~6 datasets | 100+ datasets (~4–5M clean examples) |
| Adapter training GPU | A100 40GB | A100 40GB (or RTX 3090 for QLoRA) |
| Foundation training GPU | — | **8× A100 80GB**, ZeRO-3 |
| Foundation training cost | ~$1,500 | ~$30 (5M corpus, 3 epochs) |
| Full pipeline cost | ~$5,000 | **~$35** |
| Evaluation | Basic metrics | 9 metrics with pass/fail thresholds |
| Compute dtype | float16 | bfloat16 |
