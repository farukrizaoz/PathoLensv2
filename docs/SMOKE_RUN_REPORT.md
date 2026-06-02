# Smoke Run Report — Grounded Pathology VLM (PathoLens-VLM)

**Authors:** Faruk Rıza Öz, Emir Arda Eker
**Supervisor:** Prof. Dr. Behçet Uğur Töreyin (ITU AI & Data Engineering, Capstone 2026)
**Period:** May 31 – June 2, 2026
**Status:** End-to-end pipeline validated on a 10-slide smoke subset. Three honest training runs (v1, v2, v3) recorded. Ready for full-scale.

---

## 1. Abstract

This document is the academic record of the smoke-scale validation of PathoLens-VLM, a vision-language architecture that produces spatially-grounded pathology reports from whole-slide images (WSIs). The contribution is not raw caption quality but the *grounding mechanism*: a layer-14 text→vision attention head is supervised against pixel-level BCSS tissue masks so that every generated sentence carries a verifiable spatial origin. We ran three end-to-end training cycles on 10 TCGA-BRCA slides. In doing so we identified and corrected four substantive bugs (two of which silently invalidated the original loss objective) and produced honest baseline numbers for caption quality, grounding loss convergence, and attention concentration. The pipeline is now reproducible from a single shell script, validated by a 5-phase dry-run gate, and observable through a live Streamlit demo + WandB dashboard.

---

## 2. Architecture (as implemented)

```
WSI (.svs, TCGA-BRCA)
  └─ Tissue segmentation (Otsu on HSV-saturation thumbnail @1.25×)
     └─ Patch extraction (448×448 @20×, level-0 coordinates retained)
        └─ CONCHv1.5 ViT-L/16 patch encoder            [FROZEN, 304 M params]
            └─ L2-normalised 768-dim patch tokens (N_patches per slide)
                └─ Max-subsample to N_v ≤ 1024
                    └─ Linear Adapter (768 → 3072)      [TRAINABLE, ~5 M]
                        └─ LLaVA-style prepend to chat tokens
                            └─ Llama-3.2-3B-Instruct    [FROZEN base]
                              + LoRA r=16 on q/k/v/o    [TRAINABLE, ~10 M]
                                └─ Logits (B, N_v + T, 128k)
                                └─ text→vision attention from layer 14
                                  → per-sentence (N_v,) attention distribution
                                  → HL7 FHIR DiagnosticReport JSON
```

**Trainable surface:** 15 M params / 3.21 B total = **0.46 %**.
**Memory:** A6000 48 GB with bf16 + 4-bit base via `bitsandbytes` + gradient checkpointing.

---

## 3. Data pipeline

### 3.1 Source datasets

| Source | Role | Notes |
|---|---|---|
| **TCGA-BRCA** (DX FFPE slides) | Visual input | 881 BRCA DX slides total. We pulled 10 for the smoke. SVS streamed via GDC `/data/{file_id}` API. |
| **SlideChat** (`General-Medical-AI/SlideChat`) | Caption supervision | Slide-aligned by DX barcode. Captions are GPT-4-generated structured reports, *not* the original TCGA pathology PDFs. 881 BRCA slides matched. |
| **BCSS** (figshare ROI masks) | Grounding supervision | Pixel-level annotations of tumor / stroma / lymphocytic / necrosis regions on a subset of TCGA-BRCA slides. 10 slides matched our smoke pool. |

### 3.2 Slide selection (`scripts/select_smoke_set.py`)

The training pool is the **intersection** of: (a) DX (FFPE) TCGA-BRCA slides on GDC, (b) slides with a SlideChat caption, (c) slides with a BCSS pixel mask. Joined on the canonical DX barcode `TCGA-XX-XXXX-NN[A-Z]-NN-DX[1-9]`. Yielded 112 candidate slides; we picked 10 for the smoke run.

### 3.3 Manifest patch (`scripts/fix_manifest_fileids.py`)

BCSS-embedded GDC UUIDs are stale (older than current GDC indexing). We re-query GDC `/files` with a BRCA + DX + open-access filter and rewrite the manifest's `file_id` for each barcode. Without this step every download returned 61-byte JSON error blobs.

### 3.4 Embedding precompute (`patholens.data.precompute_embeddings`)

For each slide:
1. Tissue mask at 1.25× thumbnail (Otsu on HSV saturation, fallback morphological).
2. Patch extraction at 20× → `(N_patches, 3, 448, 448)` + `(N_patches, 2)` level-0 coordinates.
3. CONCHv1.5 (bundled inside TITAN, `MahmoodLab/TITAN.return_conch()`) → L2-normalised 768-d embeddings.
4. Optional TITAN slide-aggregation skipped (API drift; not used downstream).
5. CONCH v1 (512-d) skipped (HF access blocked; only used for pseudo-GT fallback which never fires for BCSS-covered smoke slides).

HDF5 schema per slide:
| Dataset | Shape | Purpose |
|---|---|---|
| `patch_embeddings_v15` | `(N, 768)` fp16 | Vision tokens fed to the LLM |
| `coordinates` | `(N, 2)` int32 | Level-0 (x, y) — used to map attention back to pixels |
| `tissue_mask_lowres` | `(H, W)` uint8 | Thumbnail-resolution tissue silhouette for the demo overlay |

**Wall-clock cost:** ~5–8 min per slide on RTX A6000; 10 slides = 46 min total.

**Sanity (10/10 slides):** N_patches ∈ [3 719, 27 340]; embedding mean L2-norm = 1.000 ± 0; no NaN/Inf.

### 3.5 BCSS patch masks (`scripts/06_prepare_bcss_masks.py`)

For each slide, intersect every patch's level-0 bounding box with the figshare PNG mask. Output: per-patch coverage values in [0, 1] for each of 5 channels (TUMOR, STROMA, LYMPH, NECROSIS, ROI). Written to `data/processed/bcss_patch_masks/{slide_id}.h5`.

**ROI coverage per slide:** 49–440 patches (out of 3 719–27 340 total).

### 3.6 SlideChat → instruction set (`scripts/07_convert_slidechat.py` + `make_smoke_instruction.py`)

SlideChat ships as raw `{id, image, conversations}` records. Conversion:
1. BRCA-only filter (image path contains `/BRCA/`).
2. Group conversations by DX barcode.
3. Pick the *most report-like* GPT answer as the slide caption (keywords: `describe`, `summary`, `findings`, `overview`, `report`; fallback: longest answer).
4. Remaining (human, gpt) turns become `vqa_pairs` for future use.
5. Deterministic 70/15/15 split *by slide* (no slide leaks across train/val/test).

Smoke instruction set after `make_smoke_instruction.py`: **8 train / 2 val / 10 test**.

---

## 4. Model formulation

### 4.1 Forward pass (LLaVA-style prepend)

For a single training example with `N_v` vision tokens and `T` text tokens:

$$ \mathbf{V} = \text{Adapter}(\mathbf{X}_{\text{patches}}) \in \mathbb{R}^{N_v \times 3072} $$
$$ \mathbf{E} = \text{LLM-embed}(\text{input\_ids}) \in \mathbb{R}^{T \times 3072} $$
$$ \mathbf{Z} = [\mathbf{V}; \mathbf{E}] \in \mathbb{R}^{(N_v + T) \times 3072} $$

LLM consumes $\mathbf{Z}$ via `inputs_embeds`. Causal attention mask is all-ones for vision positions plus the original text attention mask. Labels are `[-100^{N_v}; \text{text-labels}]` so vision positions never appear in the LM loss.

### 4.2 Layer-14 attention extraction

For grounding supervision we extract the attention block from a fixed mid-layer:

$$ A^{(14)} \in \mathbb{R}^{B \times H \times (N_v+T) \times (N_v+T)} $$
$$ A_{\text{txt→vis}} = \text{mean}_{H}\, A^{(14)}_{:, :, N_v:, :N_v} \in \mathbb{R}^{B \times T \times N_v} $$

For each sentence with token span $[s_i, e_i]$:

$$ a_i = \text{mean}_{t \in [s_i, e_i]}\, A_{\text{txt→vis}}[0, t, :] \in \mathbb{R}^{N_v} $$

This $a_i$ is the model's per-sentence claim about *where it looked*.

### 4.3 Combined loss

$$ \mathcal{L} = \mathcal{L}_{\text{cap}} + \lambda_g \cdot \mathcal{L}_{\text{ground}} + \lambda_f \cdot \mathcal{L}_{\text{faith}} $$

**Caption loss.** Standard shifted causal-LM cross-entropy with `ignore_index=-100`. Vision-prefix positions excluded by padding labels with `-100`.

**Grounding loss.** KL between per-sentence normalised attention and BCSS-derived target $g_i$:

$$ \hat a_i = a_i / \sum_k a_{i,k},\qquad \mathcal{L}_{\text{ground}} = \frac{1}{S}\sum_i \mathrm{KL}\big(g_i \,\|\, \hat a_i\big) $$

`build_target` routes per sentence: BCSS mask if the sentence mentions a concept keyword *and* the slide has BCSS coverage above threshold, else uniform fallback (the original design used a CONCH-v1 pseudo-GT here; access is blocked, so we degrade to uniform).

**Faithfulness regulariser.** Entropy of the per-sentence attention distribution (lower = peakier):

$$ \mathcal{L}_{\text{faith}} = \frac{1}{S}\sum_i -\sum_k \hat a_{i,k}\log\hat a_{i,k} $$

### 4.4 Loss-weight warmup

`grounding_warmup_epochs` ∈ ℝ. While `epoch < warmup`, both $\lambda_g$ and $\lambda_f$ are set to 0 (caption-only). After warmup the user-supplied lambdas are restored. Lets the adapter learn to read vision tokens before grounding/faithfulness compete for gradient budget.

### 4.5 Optimizer

AdamW with **two parameter groups**:
- Linear adapter: `lr = 1e-4` (higher; it's a randomly-initialised bridge)
- LoRA params: `lr = 5e-5`

Cosine schedule with 2–4 warmup steps. `max_grad_norm = 1.0`. bf16 mixed precision via HF Trainer; 4-bit base via bitsandbytes; gradient checkpointing enabled inside the LoRA wrapper.

---

## 5. Experiments

Three runs on the same 10-slide BRCA pool (8 train / 2 val / 10 test). Compared head-to-head to expose the impact of the loss-fix audit (§6).

### 5.1 Run table

| Run | Date | Config | Epochs | $\lambda_g$ | $\lambda_f$ | Warmup | Notes |
|---|---|---|---|---|---|---|---|
| **v1** | 2026-06-02 10:48 | `grounding_smoke.yaml` | 4 | 0.3 | 0.1 | none | Pre-audit. Two losses silently broken. |
| **v2** | 2026-06-02 12:13 | `grounding_smoke.yaml` | 4 | 0.3 | 0.1 | none | Post-audit. Same weights; reveals the true regime. |
| **v3** | 2026-06-02 16:14 | `grounding_smoke_v2.yaml` | 8 | **0.1** | **0.01** | **2 epochs** | Retuned. Honest baseline. |

### 5.2 Training trajectory

#### Caption loss (lower is better)
| Run | Epoch 1 | Epoch 4 | Epoch 8 | eval_loss |
|---|---|---|---|---|
| v1 | 6.9 | **1.86** | — | 2.55 |
| v2 | 11.8 | 5.95 | — | 4.37 (broken at save; eval from trainer) |
| v3 | 11.8 (frozen by warmup) | 2.0 | **1.4 – 2.2** | **2.32** |

#### Grounding loss (lower is better)
| Run | Epoch 1 | Epoch 4 | Epoch 8 | Interpretation |
|---|---|---|---|---|
| v1 | ~1e-8 | ~1e-8 | — | **Loss was a no-op** — double-softmax made both distributions ~uniform → KL ≈ 0. |
| v2 | 5–35 (spiky) | 2–35 | — | Real magnitude but unstable; with $\lambda_g{=}0.3$ it dominates total loss. |
| v3 | 0.007 (gated off) | 1.7 (after warmup) | **0.5–0.7** | Real, smooth, decreasing **5×** during epochs 2–8. |

#### Faithfulness loss (lower = peakier attention)
| Run | Final value | Interpretation |
|---|---|---|
| v1 | 6.93 constant | **Bug**: `softmax(small_values)` → uniform → entropy fixed at log(N_v). |
| v2 | 6.80 (brief 4.8 dips) | Mechanism responsive but model can't go peaky in 4 epochs. |
| v3 | 6.92 | Still high — model does not learn to concentrate on 8 slides; problem is data scale, not a bug. |

### 5.3 Held-out test metrics

#### Caption quality (10 test slides, BLEU-4 + ROUGE-L)

| Run | BLEU-4 (corpus) | ROUGE-L (corpus) | Per-slide BLEU-4 range |
|---|---|---|---|
| v1 (broken) | 0.0566 | 0.1916 | 0.024–0.122 |
| v2 (post-audit, untuned) | **0.0055** | **0.0263** | 0.001–0.014 |
| v3 (retuned + warmup) | **0.0566** | **0.1808** | 0.024–0.121 |

**Reading.** v1's BLEU was inflated because grounding was a no-op — the optimizer spent 100 % of capacity on caption. v2 (real grounding at $\lambda_g{=}0.3$) crushed caption because grounding magnitudes dominated total loss. v3 (retuned weights + warmup) recovers v1's caption quality while *also* training a real grounding signal — the honest baseline.

**Interpretation of the absolute numbers.** A BLEU-4 of 0.057 on 8 training slides against a synthetic GPT-4 caption target is *expected and uninformative*. The model has learned report structure (~19 % LCS overlap, ROUGE-L = 0.18) but not slide-specific content. This is also why generated reports contain `[insert findings]` placeholders — the base Llama-3.2-3B's RLHF prior emits a templated pathology report shape when uncertain, and 32 gradient steps cannot override it.

#### Grounding quality (concentration metric, v3 only)

For every generated sentence we compute the (N_v = 1024) attention distribution and report:
- **top-k% mass** — sum of the top k% largest values (higher = peakier)
- **entropy ratio** — entropy / log(N_v) (1.0 = uniform, 0.0 = single-patch)

| Metric | v3 value | Random (uniform) | Interpretation |
|---|---|---|---|
| top-1% mass | 0.013 | 0.010 | barely above uniform |
| top-5% mass | 0.064 | 0.050 | barely above uniform |
| top-10% mass | 0.124 | 0.100 | mildly peaky |
| entropy ratio | 0.999 | 1.000 | essentially uniform |

**Reading.** After 8 epochs on 8 slides, the model's attention is only marginally non-uniform. This is consistent with $\mathcal{L}_{\text{faith}}$ staying at ~log(N_v) throughout training: with $\lambda_f{=}0.01$ and only ~32 supervised gradient steps post-warmup, the entropy penalty does not have enough budget to bend the attention distribution. The grounding KL *does* decrease (5× over training), so the model is learning to *match the shape* of the BCSS target, but the absolute magnitude of the attention values on the supervised patches is small. Both losses are functioning correctly; they need more data and longer training to manifest as visible attention peakiness. This is the central observation that motivates the full-scale run.

---

## 6. Audit and bug history

Discovered and fixed during this smoke campaign. Four were correctness bugs; three were environment/integration regressions. All have a dry-run check that prevents regression.

| # | Symptom | Root cause | Fix | File |
|---|---|---|---|---|
| 1 | Faithfulness loss stuck at 6.931 = log(1024) for entire training | `torch.softmax` applied to per-patch attention values of magnitude ~1e-3 → distribution collapses to ~1/N_v → entropy maximal | Renormalise by sum (proper probability), not softmax. Pass raw attention into the regulariser; it normalises internally. | `training/losses.py` |
| 2 | Grounding loss permanently ≈ 1e-8 ("converged"); test attention indistinguishable from random | `gt_attn` was already a probability distribution from `build_target`, but `GroundingLoss` applied a second `softmax` with temperature, smoothing it back to ~uniform; `KL(uniform ‖ uniform) ≈ 0` | Normalise both distributions once with sum-to-1; apply temperature sharpening inside the loss in log-space. | `training/losses.py` |
| 3 | `RuntimeError: mat1 and mat2 must have the same dtype/device` on first forward of `from_pretrained` checkpoint | `.load()` left adapter on CPU/fp32 while LLM is on cuda/bf16. `from_pretrained` similarly didn't cast. | (a) `.load()` moves adapter to LLM device; (b) `from_pretrained` casts adapter to bf16; (c) `forward`/`generate` cast `vision_embeds` to `text_embeds.dtype` before `torch.cat`. | `models/grounded_vlm.py`, `models/adapter.py` |
| 4 | `ValueError: Expected input batch_size (1312) to match target batch_size (288)` in CE loss | LLM logits are over $(N_v+T)$ positions but labels are only over $T$. | Pad labels with `-100` for the vision prefix in `compute_loss`. | `training/train.py` |
| 5 | `safetensors` raises "shared tensors" when saving Llama (`embed_tokens.weight` shares storage with `lm_head.weight`) | Llama-3.2-3B ties input/output embeddings; safetensors refuses to write shared storage. | `safe_serialization=False` in both `llm.save_pretrained` and `TrainingArguments(save_safetensors=False)`. | `models/grounded_vlm.py`, `training/train.py` |
| 6 | `GatedRepoError` on `MahmoodLab/CONCH` blocks training start | HF account lacks CONCH v1 access. v1 was used for pseudo-GT fallback in `build_target`. | Made v1 loading optional in both precompute and training; `build_target` returns uniform fallback when v1 is `None`. | `data/precompute_embeddings.py`, `training/train.py`, `training/grounding_targets.py` |
| 7 | `transformers 5.9` incompatible with TITAN modeling code (`'Titan' object has no attribute 'all_tied_weights_keys'`) | TITAN modeling files pre-date the v5 PreTrainedModel API. | Pin `transformers==4.46.3` in `pyproject.toml`. | `pyproject.toml` |

**Impact ranking.** Bugs #1 and #2 silently invalidated the central scientific objective: the loss the paper would otherwise claim to optimise was effectively `0 · L_grounding + 0 · L_faithfulness ≈ L_caption`. Without this audit we would have reported a publishable BLEU number that was a side-effect of training caption-only.

---

## 7. Verification gate

`scripts/dry_run_pipeline.py` is a 5-phase pre-flight that exercises the full pipeline on 1 slide in <2 min:

1. **H5 schema check** — required datasets present; embeddings L2-normalised; no NaN/Inf; coordinate shape.
2. **BCSS mask check** — file exists; ROI coverage non-zero.
3. **Dataset + collate** — `GroundingSlideDataset` returns the right keys; `grounding_collate_fn` produces a usable batch.
4. **Single forward + combined loss** — exercises adapter, LoRA, attention extraction, all three losses, and backward. Asserts losses are finite. Verifies the **faithfulness regulariser is responsive** by injecting a peaky distribution and confirming entropy drops far below log(N_v).
5. **Checkpoint round-trip** — loads via `GroundedVLM.from_pretrained`, runs `generate_with_attention`, asserts text non-empty and every per-sentence attention vector sums to 1 with no NaN/Inf.

The chained runner `scripts/retrain_v2.sh` runs the dry-run as the first step, so any regression of fixes #1–#7 fails the gate before consuming GPU time on training.

---

## 8. Proof-of-concept demo

`src/patholens/app/streamlit_app.py` is a single-page Streamlit application:

- **Sidebar:** slide dropdown (10 cached H5s), instruction text area, max-new-tokens slider, *Generate* button.
- **Tab 1 — Report:** the generated multi-sentence text.
- **Tab 2 — Grounding:** per-sentence radio selector → magma-colored attention heatmap overlaid on the slide's tissue silhouette, plus "top-10 % patch mass" concentration metric.
- **Tab 3 — FHIR:** HL7 R4 DiagnosticReport JSON via the existing `reporting/fhir.py`, with download button.
- **Tab 4 — Debug:** patch counts, vision-token count, checkpoint id, device.

Inference path: `app/inference.py:generate_with_attention` calls `model.generate(...)` then a second `forward(output_attentions=True)` on `[prompt || generated]` and slices the layer-14 text→vision block over each sentence's token span. (HF `generate(inputs_embeds=...)` does not return attentions, so the two-pass approach is unavoidable.)

Tunnel from local laptop: `ssh -L 8501:localhost:8501 …`, open `http://localhost:8501`.

---

## 9. Limitations and threats to validity

1. **Caption target is synthetic.** SlideChat captions are GPT-4 outputs, not the original TCGA pathologist reports. BLEU/ROUGE measure proximity to GPT-4's writing style, not clinical correctness. Real TCGA reports exist as scanned PDFs but are per-case (not per-slide) and require OCR; out of scope for this campaign.
2. **Smoke training pool is tiny.** 8 train slides × 8 epochs = 32 gradient updates per loss term. Insufficient to override Llama-3.2-3B's RLHF prior; cannot produce slide-specific content. The numbers above are pipeline-validation baselines, not claims of model quality.
3. **CAMELYON16 pointing game not yet run.** The concentration metric is a cheap stand-in; the publishable grounding metric requires CAMELYON16 pixel masks on held-out slides.
4. **CONCH v1 access blocked.** Non-BCSS slides currently receive a uniform fallback for grounding supervision. For the smoke run all slides have BCSS coverage so this never fires, but full-scale training over the 881-slide BRCA pool will need either CONCH v1 access or a restricted training set.
5. **Faithfulness regularizer does not bend attention at this data scale.** Mechanism verified responsive (synthetic peaky input gives entropy 2.30), but the gradient signal is too weak relative to caption loss with $\lambda_f{=}0.01$ and 8 slides.
6. **TITAN slide-token aggregation API is broken in current transformers.** Wrapped in try/except; slide tokens are not used by the model. If TITAN's mid-level slide representation is needed in future work it requires either a transformers downgrade or a manual reimplementation of `encode_slide_from_patch_features`.

---

## 10. Reproduction recipe

On RunPod (RTX A6000 48 GB, CUDA 12.4 driver):

```bash
# 0. Restore environment
cd /workspace/patholens-vlm
uv sync                         # transformers 4.46.3, torch 2.5.1+cu124, streamlit ≥1.40
source .env                     # HF_TOKEN, WANDB_API_KEY

# 1. Validate before any long run
uv run python scripts/dry_run_pipeline.py \
    --config configs/grounding_smoke_v2.yaml

# 2. Reproducible train + eval chain
bash scripts/retrain_v2.sh      # writes results/<run>_caption.json + _grounding.json
                                # final flag: /workspace/logs/RETRAIN_OK

# 3. Launch demo (optional)
bash scripts/run_app.sh         # Streamlit on :8501
ssh -L 8501:localhost:8501 …    # tunnel from laptop
```

Artifacts of interest on `/workspace`:

```
data/raw/tcga_brca/                       10× .svs (~11 GB)
data/processed/embeddings/tcga_brca/      10× .h5  (patch embeddings + coords + tissue mask)
data/processed/bcss_patch_masks/          10× .h5  (per-channel BCSS coverage)
data/processed/slideinstruction_smoke/    train.json / val.json / test.json
checkpoints/grounded_smoke_v3_20260602_161431/final/   v3 trained model
results/grounded_smoke_v3_20260602_161431_caption.json   BLEU + ROUGE
results/grounded_smoke_v3_20260602_161431_grounding.json concentration metric
wandb/                                     run dirs for all three runs
```

---

## 11. Next steps

1. **Full-scale data scope decision.** Three candidates:
   (a) ~50 BCSS-covered slides, lowest precompute cost (~4 GPU-hr);
   (b) ~150 BCSS-covered slides, strongest grounding signal (~12 GPU-hr precompute);
   (c) ~500–800 slides, full BRCA pool, caption-only for non-BCSS slides (~40–60 GPU-hr precompute).
2. **CAMELYON16 pointing-game evaluation.** Pull CAMELYON16 tumor pixel masks; implement `evaluation/pointing_game.py`; report PG@1 / PG@5 / PG@10.
3. **Faithfulness intervention test.** Mask top-K attended patches → regenerate sentence → measure text drift. Currently a stub at `evaluation/intervention_test.py`.
4. **Faithfulness λ retune at scale.** Once corpus is larger, bump $\lambda_f$ to 0.05; the regulariser should then bend attention visibly.
5. **Held-out test set expansion.** Move to ≥30 test slides for confidence intervals on BLEU/ROUGE.
6. **CONCH v1 HF access request.** Unlocks pseudo-GT supervision for non-BCSS slides; required only for slide-pool option (c).
7. **Final write-up.** Capstone report — this document is the precursor to the methodology + results sections.

---

## 12. Acknowledgements

- MahmoodLab for CONCHv1.5 and TITAN model weights and bundled accessors.
- BCSS authors for pixel-level annotations on TCGA-BRCA.
- The SlideChat team for structured WSI-caption pairs.
- Meta for Llama-3.2-3B-Instruct.
- RunPod for the A6000 environment.

## 13. References

- **TITAN:** https://huggingface.co/MahmoodLab/TITAN — paper: https://arxiv.org/abs/2411.19666
- **CONCHv1.5:** https://huggingface.co/MahmoodLab/conchv1_5
- **Llama-3.2-3B-Instruct:** https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- **SlideChat (reference, not baseline):** https://arxiv.org/abs/2410.11761
- **MedGround (grounding methodology reference):** https://arxiv.org/abs/2601.06847
- **Direct Visual Grounding via KL Attention Loss:** https://arxiv.org/abs/2511.12738
