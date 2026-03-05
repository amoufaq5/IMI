# IMI Medical AI — Investor Demo Script

## Overview

**IMI** is a 5-layer medical AI platform powered by Mistral 7B with 6 domain-specific LoRA adapters,
deterministic safety guardrails, and a knowledge-graph-backed verification system.

**Key differentiators:**
- **5-Layer Safety Architecture** — Memory → Knowledge Graph → Rule Engine → LLM → Verifier
- **Mistral 7B** base with full fine-tuning per user type (doctor, patient, student, researcher, hospital, pharma)
- **Zero hallucination tolerance** — every LLM output verified against knowledge graph + guidelines
- **DPO safety alignment** — model trained to prefer safe responses via Direct Preference Optimization
- **Sub-200ms inference** via vLLM with LoRA hot-swapping

---

## Demo Flow (15 minutes)

### 1. Patient Triage (3 min)
**Scenario:** A patient describes symptoms of chest pain.

```
User (patient): I've been having chest pain for the last 2 hours, 
it feels like pressure and my left arm is tingling.
```

**Expected behavior:**
- ⚡ InputGuardrail detects EMERGENCY signal instantly (< 5ms)
- 🚨 Emergency banner prepended: "Call 911 immediately"
- 🧠 LLM generates empathetic response with appropriate urgency
- ✅ Verifier confirms emergency referral is present
- 📋 Full audit trail logged

**Talking points:**
- Pattern-matching guardrails catch emergencies BEFORE the LLM even runs
- Deterministic rule engine handles triage — not the LLM
- No hallucination possible for emergency protocols

### 2. Crisis Detection (2 min)
**Scenario:** User expresses suicidal ideation.

```
User: I've been thinking about ending my life. Nothing feels worth it.
```

**Expected behavior:**
- 🛑 InputGuardrail triggers CRISIS signal — LLM is COMPLETELY BYPASSED
- 📞 Immediate display of crisis resources (988 Lifeline, Crisis Text Line, 911)
- 🔒 No medical advice generated — only crisis support
- 📋 Audit log flags for clinical review

**Talking points:**
- The LLM NEVER sees or responds to crisis messages
- Hardcoded crisis response — cannot be jailbroken or prompt-injected
- This is the standard of care for digital health platforms

### 3. Doctor Clinical Decision Support (3 min)
**Scenario:** A physician asks about drug interactions.

```
User (doctor): My patient is on warfarin and I'm considering adding 
fluconazole for a fungal infection. What should I watch for?
```

**Expected behavior:**
- 💊 Clinical Decision adapter loaded automatically
- ⚠️ Contraindication checker flags warfarin + fluconazole interaction
- 📚 Knowledge graph retrieves relevant drug interaction data
- 🔬 LLM synthesizes clinical guidance with INR monitoring recommendations
- ✅ Verifier checks against clinical guidelines

**Talking points:**
- Adapter switching is automatic based on user role
- Drug interaction checking is DETERMINISTIC (rule engine) — not LLM guesswork
- Doctor receives evidence-based, citation-backed response

### 4. Medical Education (2 min)
**Scenario:** A medical student studying for USMLE.

```
User (student): Explain the pathophysiology of diabetic ketoacidosis 
and the key management steps.
```

**Expected behavior:**
- 📖 Education adapter loaded
- 🎓 Structured, teaching-oriented response with key concepts highlighted
- 📊 Step-by-step management protocol
- ⚠️ Appropriate disclaimers for clinical application

### 5. Safety Boundary Demo (2 min)
**Scenario:** Testing scope enforcement.

```
User: Can you write me a prescription for amoxicillin?
```

**Expected behavior:**
- 🚫 InputGuardrail detects SCOPE_VIOLATION
- 📝 Clear explanation: "I cannot write prescriptions..."
- 🔄 Redirects user to appropriate resource

### 6. Architecture Walkthrough (3 min)
Show the 5-layer pipeline visually:

```
Query → [Layer 0: Memory] → [Layer 1: Knowledge Graph] → [Layer 2: Rule Engine]
                                                                    ↓
                                                        [Layer 3: LLM + Adapter]
                                                                    ↓
                                                        [Layer 4: Verifier]
                                                                    ↓
                                                            Final Response
```

**Talking points:**
- The LLM is layer 3 of 5 — it NEVER makes safety decisions alone
- Every response is verified before reaching the user
- Full HIPAA-compliant audit trail for every interaction
- Knowledge graph provides grounding — reduces hallucination

---

## Technical Specs

| Component | Specification |
|-----------|--------------|
| Base Model | Mistral 7B-Instruct (Apache 2.0) |
| Training | Full fine-tuning, BFloat16 compute |
| Adapters | 6 LoRA adapters (r=32, α=64) |
| Training Data | 40+ open medical datasets, 3M+ examples |
| Safety | DPO alignment + regex guardrails + rule engine + verifier |
| Inference | vLLM with LoRA hot-swap, <200ms p95 latency |
| Infrastructure | 6×A100 80GB (training), 2×A100 (inference) |
| Compliance | HIPAA-ready, AES-256-GCM encryption, full audit logging |

## Budget Summary

| Phase | Cost | Duration |
|-------|------|----------|
| Data collection & processing | $0 (open datasets) | Week 1 |
| Foundation training (Mistral 7B) | ~$1,500 | Week 1-2 |
| DPO safety alignment | ~$200 | Week 2 |
| Adapter training (6 adapters) | ~$2,000 | Week 2-3 |
| Evaluation & iteration | ~$300 | Week 3 |
| Inference hosting (monthly) | ~$1,000/mo | Ongoing |
| **Total MVP** | **~$5,000** | **3 weeks** |

## Competitive Advantages

1. **Open-source base** — No vendor lock-in, Apache 2.0 licensed
2. **Multi-layer safety** — Not just an LLM wrapper; deterministic safety that can't be jailbroken
3. **6 specialized adapters** — Purpose-built for each user type, not one-size-fits-all
4. **Knowledge graph grounding** — Responses backed by verified medical knowledge
5. **Full audit trail** — Every query, response, and safety decision logged for compliance
6. **Cost-efficient** — Full fine-tuning on H100 GPUs, vLLM inference at scale

---

## Q&A Preparation

**Q: How do you prevent hallucinations?**
A: 5-layer defense: (1) Knowledge graph grounds responses in verified data, (2) Rule engine enforces deterministic safety rules, (3) DPO alignment trains the model to prefer safe responses, (4) Output guardrails catch overconfident language, (5) Verifier cross-checks against guidelines.

**Q: Is this HIPAA compliant?**
A: Architecture is HIPAA-ready with AES-256 encryption, full audit logging, role-based access, and no patient data in training. BAA-ready for deployment.

**Q: How does this compare to GPT-4 medical?**
A: GPT-4 is a black box with no safety layers. IMI wraps a capable open-source LLM in 4 additional safety layers. We control the model, the safety, and the audit trail.

**Q: What's the go-to-market?**
A: Phase 1 (MVP): Doctor + Patient adapters → B2B to clinics/telehealth. Phase 2: Education + Research → medical schools. Phase 3: Hospital + Pharma → enterprise.
