# Architecture

Detailed architectural decisions, design rationale, and hyperparameter choices.

## Full Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  WSI input (.svs, ~50K × 50K pixels @20×)                           │
│       │                                                             │
│       ▼                                                             │
│  Tissue Segmentation (Otsu @ 1.25× / GrandQC fallback)              │
│       │                                                             │
│       ▼                                                             │
│  Patch extraction: 448×448 RGB tiles @20× from tissue regions       │
│       │ (~3K–7K patches per slide)                                  │
│       ▼                                                             │
│  CONCHv1.5 ViT-L patch encoder  [FROZEN]                            │
│       │ patch_embeddings_v15: (N_patches, 768)                      │
│       │                                                             │
│       ├──▶ TITAN Slide Encoder [FROZEN]  ← stored, not used in LLM  │
│       │       slide_tokens: (M, 768) — future use only              │
│       │                                                             │
│       ▼  ← vision tokens for LLM are CONCHv1.5 patches, NOT TITAN  │
│  Linear Adapter  [TRAINABLE, ~5M params]                            │
│       │ 768 → 3072 (Llama hidden dim)                               │
│       │ adapter_tokens: (N_v, 3072)   N_v ≤ max_vision_tokens=1024  │
│       ▼                                                             │
│  Llama-3.2-3B-Instruct + LoRA (r=16) [TRAINABLE, ~10M params]       │
│       │                                                             │
│       │ Input sequence (LLaVA-style prepend):                       │
│       │   [adapter_tokens (N_v)] + [instruction+response tokens (T)]│
│       │   Total: (N_v + T) tokens in one causal attention window    │
│       │                                                             │
│       ├───▶ Generated structured report                             │
│       │                                                             │
│       └───▶ Self-attention extraction (layer 14, avg over heads)    │
│              attentions[14]: (B, H, N_v+T, N_v+T)                  │
│              text→vision slice: [:, :, N_v:, :N_v] → (B, H, T, N_v)│
│              mean over heads: → (B, T, N_v)                         │
│              aggregate over sentence span: → (S, N_v)               │
│              ▼                                                      │
│           Per-sentence grounding heatmap                            │
│           (1:1 with input patches, no mapping step needed)          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Critical Design Decision: Vision Tokens are CONCHv1.5 Patches, NOT TITAN Slide Tokens

**Problem:** The original plan (pre-implementation) described TITAN slide tokens as the LLM vision input. TITAN produces M slide-level tokens via hierarchical attention over N patches, where M ≠ N and the mapping from slide tokens back to individual patches is non-trivial.

**Why this breaks grounding:** The grounding ground truth (BCSS masks, CONCH pseudo-GT) is defined at the **patch level** — a distribution over N patches. If the LLM attends to M TITAN slide tokens, we have no 1:1 correspondence between LLM attention weights and patch-level GT.

**Resolution:** Use CONCHv1.5 patch embeddings (same 768-dim) directly as LLM vision tokens, capped at `max_vision_tokens=1024`. The adapter projects from 768 → 3072 (Llama hidden dim) regardless of source.

| Property | CONCHv1.5 patches (CHOSEN) | TITAN slide tokens |
|---|---|---|
| LLM input dim | 768 ✓ | 768 ✓ |
| Count | N patches (capped) | M slide tokens |
| 1:1 with GT | YES ✓ | NO ✗ |
| TITAN still precomputed? | Yes (stored in HDF5, future use) | — |

**TITAN is still precomputed** and stored in the HDF5 cache. It may be used for a global slide context token in a future ablation, but is not in the current training path.

---

## Component Decisions

### Tissue Segmentation: Otsu + GrandQC (Fallback)

**Otsu (default):** Fast, HSV saturation thresholding on 1.25× thumbnail. Sufficient for most clean TCGA slides.

**GrandQC (fallback):** If Otsu tissue area < threshold, GrandQC pretrained tissue segmentation is used. GrandQC is publicly available on HuggingFace.

### Patch Size: 448×448 @20×

CONCHv1.5 was trained on 448×448 patches. TITAN also requires v1.5 features extracted at 448. Stride = 448 (no overlap), minimizing patch count.

Typical TCGA-BRCA slide: ~3K–7K patches.

### Patch Encoder: CONCHv1.5 (ViT-L)

- Compatible with TITAN (TITAN only accepts v1.5 features)
- 768-dim embeddings, same as TITAN slide tokens → same adapter applies to both
- **Vision-only** — no text encoder. For pseudo-GT, CONCH v1 (512-dim, shared vision+text space) is used separately
- **Frozen.** Embeddings are precomputed → HDF5 cache → no GPU cost at training time

**License:** CC-BY-NC-ND 4.0, research-only.

### Slide Encoder: TITAN (Frozen, Stored but not in current LLM path)

TITAN is precomputed and stored in the HDF5 cache. Its output slide tokens are available for future ablations (e.g. prepending a single global slide token before patch tokens). Not in the current training loop.

### Adapter: Linear Projection (LLaVA-Style)

```python
class LinearAdapter(nn.Module):
    def __init__(self, in_dim=768, out_dim=3072):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
```

LLaVA-1.5 showed linear ≈ Q-Former performance. 5M parameters, fast, easy to debug.

### LLM: Llama-3.2-3B-Instruct

| Spec | Value |
|------|-------|
| Parameters | 3.21B |
| Hidden dim | 3072 |
| Layers | 28 |
| Attention heads | 24 |
| Context window | 128K (effective use: ~2K) |
| License | Meta Llama 3.2 Community |

**LoRA configuration:**
```yaml
r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
bias: "none"
task_type: "CAUSAL_LM"
```

Trainable params: ~10M LoRA + ~5M adapter = ~15M total.

**4-bit quantization (RunPod):** bitsandbytes NF4 + double_quant + bf16 compute dtype. Disabled on CPU (local testing).

**Why 3B:** 3B + 4-bit + LoRA fits in ~8 GB VRAM; 2× faster than 7B; LoRA fine-tuning largely closes the 3B vs 7B gap in domain-specific captioning.

---

## Loss Design

```
L_total = L_caption + λ_g · L_grounding + λ_f · L_faithfulness
```

Defaults: `λ_g = 0.3, λ_f = 0.1`. Five ablation configs cover a 2×2 grid of (grounding_source × loss_type) plus a caption-only baseline.

### Five Ablation Configs

| Config | λ_g | Grounding source | Loss type |
|---|---|---|---|
| `caption_baseline.yaml` | 0 | — | — |
| `grounding_v1.yaml` | 0.3 | CONCH pseudo only (KL) | KL divergence |
| `grounding_v1_cosine.yaml` | 0.3 | CONCH pseudo only | cosine distance |
| `grounding_v2.yaml` | 0.3 | BCSS + CONCH hybrid | KL divergence |
| `grounding_v2_cosine.yaml` | 0.3 | BCSS + CONCH hybrid | cosine distance |

### L_caption — Standard LM Cross-Entropy

```python
# Instruction + vision token positions are masked with -100
loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
```

Ground truth: SlideInstruction captions (filtered to 151 BCSS-overlap TCGA-BRCA slides).

### L_grounding — Sentence-Patch Alignment

For each generated sentence, the model's attention distribution over vision tokens should match the expected patch-level ground truth distribution.

**Ground truth construction (hybrid router in `grounding_targets.py`):**

1. Detect concept in sentence (tumor / stroma / lymph / necrosis) via keyword matching
2. If `grounding_source == "bcss_hybrid"`:
   - Check if BCSS masks are available for this slide
   - Check ROI coverage fraction ≥ `MIN_BCSS_COVERAGE_FRACTION` (0.2)
   - If concept matched AND above threshold: use normalised BCSS mask channel → `(N_patches,) float`
3. Fallback (or if `grounding_source == "conch_only"`): CONCH v1 pseudo-GT:
   ```python
   sentence_emb = conch_v1.encode_text(sentence)       # (512,)
   patch_emb_v1 = conch_v1_patch_embeddings             # (N_patches, 512)
   gt_attn = softmax(cosine_sim(sentence_emb, patch_emb_v1) / temperature)
   ```

**Two loss variants (selectable via config `loss.loss_type`):**

```python
# KL divergence (baseline)
def kl_grounding_loss(generated_attn, gt_attn):
    return F.kl_div(F.log_softmax(generated_attn, dim=-1), gt_attn, reduction='batchmean')

# Cosine distance (ablation — softer for noisy teacher)
def cosine_grounding_loss(generated_attn, gt_attn):
    return 1.0 - F.cosine_similarity(generated_attn, gt_attn, dim=-1).mean()
```

### Attention Extraction (Layer 14)

```python
# attentions[layer]: (B, n_heads, N_v+T, N_v+T)
layer_attn = outputs.attentions[self.attn_layer]       # default: layer 14
text_to_vision = layer_attn[:, :, N_v:, :N_v]         # (B, H, T, N_v)
result["text_to_vision_attn"] = text_to_vision.mean(dim=1)  # avg heads → (B, T, N_v)

# In compute_loss: aggregate over sentence token span
sent_attn = out["text_to_vision_attn"][b_idx, span_start:span_end, :].mean(dim=0)  # (N_v,)
```

**Layer selection rationale:** Layer 14 of 28 (mid-network) has been reported in LLaVA literature (Kang et al. CVPR 2025 "Few Attention Heads Suffice") as grounding-faithful for similar architectures. This will be validated visually in Week 3; config `attn_layer` allows sweeping other layers.

### L_faithfulness — Attention Concentration Penalty

```python
# Entropy of each sentence's attention distribution → minimize to force concentration
entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1)
return entropy.mean()
```

Prevents the model from spreading attention uniformly across all patches.

---

## Sequence Layout and Label Masking

```
Input:  [vision_tokens (N_v)]  [instruction_tokens (T_inst)]  [response_tokens (T_resp)]
Labels: [-100 × N_v         ]  [-100 × T_inst              ]  [response_ids (T_resp)   ]
Mask:   [1 × N_v            ]  [1 × T_inst                 ]  [1 × T_resp              ]
```

Vision tokens always have label=-100 (never predict patch embeddings). Only response positions contribute to L_caption.

---

## Grounding Supervision: BCSS (Primary) + CONCH Pseudo-GT (Fallback)

**Why BCSS instead of CAMELYON16 for training supervision:**

CAMELYON16 was the original plan but was dropped for training after design review:
- Binary (tumor/non-tumor only) — cannot supervise stroma/lymph/necrosis sentences
- Organ mismatch: lymph node metastasis vs primary breast tumor histology
- Only one sentence type benefits ("tumor present") — the other 80% of report sentences have no supervision

**BCSS (Breast Cancer Semantic Segmentation, Amgad et al. 2019):**
- 151 TCGA-BRCA slides with pixel-level ROI annotations
- 22 tissue classes → 4 primary: tumor, stroma, lymphocytic infiltrate, necrosis_or_debris
- Same SVS files as our training set (exact TCGA-BRCA overlap)
- CC0 license
- Annotations are ROI-only (not full slide), combined with pseudo-GT for uncovered patches

**CAMELYON16** is retained as a **cross-dataset evaluation set** only (pointing game), providing a generalization narrative.

---

## Academic Hypotheses

| ID | Hypothesis | Measurement | Effort |
|----|-----------|-------------|--------|
| **H1** | CONCH pseudo-GT is a reliable grounding signal | Zero-shot AUC-ROC (already validated in PoC: AUC=0.608) | Done |
| **H2** | Grounding loss concentrates attention | Shannon entropy baseline vs grounded | Near-zero (3 lines in eval loop) |
| **H3** | Different clinical concept sentences attend to different spatial regions | Mean cosine distance between per-sentence attention maps | Medium — `evaluation/contrastive_separation.py` |
| **H4** | Q-Former adapter produces cleaner grounding than linear | Ablation: swap adapter → compare PG@5 and faithfulness | High — separate training run |

**Minimum presentation:** H1 + H2. H3/H4 if time allows.

**PoC reference:** `notebooks/poc_conch_grounding.py` — validated H1 on CONCHv1.5.

---

## Hyperparameters

Two sets of hyperparameters are relevant: the original ablation grid (five configs in `configs/`) and the fullscale tuned values that produced the published headline results.

| Parameter | Ablation configs | **Fullscale (published)** |
|---|---|---|
| optimizer | AdamW | AdamW |
| lr_adapter | 1e-4 | 1e-4 |
| lr_lora | 5e-5 | 5e-5 |
| weight_decay | 0.01 | 0.01 |
| warmup_steps | 100 | 8 |
| scheduler | cosine | cosine |
| batch_size | 1 | 1 |
| gradient_accumulation | 16 | **4** |
| epochs | 3 | **8** |
| mixed_precision | bf16 | bf16 |
| gradient_checkpointing | true | true |
| **lambda_grounding** | 0.3 | **0.1** |
| **lambda_faithfulness** | 0.1 | **0.05** |
| **grounding_warmup_epochs** | 0 | **2.0** |
| lora_r | 16 | 16 |
| lora_alpha | 32 | 32 |
| lora_dropout | 0.05 | 0.05 |
| target_modules | q/k/v/o_proj | q/k/v/o_proj |
| max_vision_tokens | 1024 | 1024 |
| attn_layer | 14 | 14 |

### Grounding Loss Warmup

`grounding_warmup_epochs` delays the grounding loss until caption training has stabilised. For the first N epochs only `L_caption` is active; `λ_g` and `λ_f` are linearly ramped in from 0 after the warmup period. This prevents the grounding signal (which is noisy early in training) from dominating before the LLM has learned to generate coherent pathology text.

The fullscale run used `grounding_warmup_epochs=2.0` — grounding loss entered at epoch 2 and was effective by epoch 3 (grounding loss dropped from ~11.8 to ~0.26 between epochs 2 and 5).

### Published Results (`configs/fullscale.yaml`, 50 slides)

| Metric | Value | Comparison |
|---|---|---|
| BLEU-4 (test) | **0.0559** | smoke v3: 0.0566 — held steady |
| ROUGE-L (test) | **0.1990** | smoke v3: 0.1808 — +10% |
| PG@5 (BCSS) | **0.081** | uniform: 0.019 — **4.4× lift** |
| Top-10% attn mass | **0.201** | smoke v3: 0.124 — +62% |
| Entropy ratio | **0.938** | smoke v3: 0.999 — substantially peakier |

Full run report: `docs/FULLSCALE_RUN_REPORT.md`

---

## Evaluation Metrics

### Caption Quality
| Metric | Tool | Ground Truth |
|--------|------|-------------|
| BLEU-4 | HF `evaluate` | SlideInstruction captions |
| ROUGE-L | HF `evaluate` | SlideInstruction captions |

### Spatial Grounding
| Metric | Tool | Ground Truth |
|--------|------|-------------|
| Pointing Game @K (BCSS) ✅ | `evaluation/bcss_pointing_game.py` | BCSS pixel masks — used in fullscale run |
| Attention concentration, entropy ratio ✅ | `evaluation/concentration_metric.py` | — (unsupervised) |
| Pointing Game @K (CAMELYON16) | `evaluation/pointing_game.py` | CAMELYON16 pixel masks _(not run — 150 GB dataset)_ |
| Faithfulness (intervention drop) | `evaluation/intervention_test.py` | Top-K patch masking _(planned)_ |

### Clinical Concept Accuracy
| Metric | Tool | Ground Truth |
|--------|------|-------------|
| Concept recall (tumor/stroma/lymph/necrosis) | `evaluation/concept_f1.py` | BCSS concept labels _(planned)_ |

---

## Compute Estimates (RunPod A6000 48 GB)

| Stage | Peak VRAM | Duration |
|-------|-----------|----------|
| BCSS preprocessing (151 slides) | 4 GB | 2–4 hours |
| CONCHv1.5 + CONCH v1 + TITAN embedding precompute | 24 GB | 8–12 hours |
| Caption baseline (3 epochs) | 28 GB | 10–14 hours |
| Grounding v1/v2 × KL/cosine (4 runs × 3 epochs) | 32 GB | ~15 hours each |
| Evaluation (full suite) | 16 GB | 4–6 hours |

Total: ~80–100 GPU hours.

---

## Risk Register

**Risk 1 — TITAN API (already resolved):**
TITAN not needed in current training path (CONCHv1.5 patch embeddings are vision tokens). TITAN is precomputed and stored; if its API changes, training is unaffected.

**Risk 2 — Vision token count:**
Variable N_v per slide (depends on patch count and max_vision_tokens cap). `batch_size=1` throughout; collate function keeps vision tokens as list, stacked in `compute_loss`. No padding needed for the vision dimension.

**Risk 3 — Attention extraction layer/head:**
Default is layer 14 (mid-network). `attn_layer` config param allows sweeping. Visual inspection planned in Week 3.

**Risk 4 — Grounding loss variant:**
KL divergence is the baseline. Cosine distance ablation covers the case where KL divergence is too strict for the noisy pseudo-GT teacher.

**Risk 5 — SlideInstruction caption noise:**
SlideInstruction uses GPT-4 structured captions, largely solving this problem.

**Risk 6 — BCSS TNBC subtype bias:**
BCSS 151 slides are not uniformly distributed across TCGA-BRCA subtypes. Triple-negative (TNBC) slides may be over-represented in the annotated ROIs because they tend to have more complex histology. This could bias the grounding toward TNBC-like patterns. Mitigation: monitor concept-F1 per subtype; report per-subtype pointing-game accuracy.

---

## References

- TITAN (Mahmood Lab, 2024): https://arxiv.org/abs/2411.19666
- CONCH (Nature Medicine 2024): https://www.nature.com/articles/s41591-024-02856-4
- CONCHv1.5 (HF model card): https://huggingface.co/MahmoodLab/conchv1_5
- BCSS (Amgad et al. 2019): https://academic.oup.com/gigascience/article/8/5/giz037/5481417
- SlideChat (CVPR 2025, reference): https://arxiv.org/abs/2410.11761
- LLaVA-1.5 (NeurIPS 2023): https://arxiv.org/abs/2310.03744
- Direct Visual Grounding via KL Attention Loss (2025): https://arxiv.org/abs/2511.12738
- MedGround (Jan 2026): https://arxiv.org/abs/2601.06847
- "Few Attention Heads Suffice" (Kang et al. CVPR 2025)
