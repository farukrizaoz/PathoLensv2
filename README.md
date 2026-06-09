# PathoLens-VLM

**Spatially-Grounded Vision-Language Model for Whole-Slide Breast Cancer Histopathology Report Generation**

Istanbul Technical University — AI & Data Engineering Capstone Project (2026)

Faruk Rıza Öz · Emir Arda Eker · Supervisor: Prof. Dr. Behçet Uğur Töreyin

---

## What this is

A vision-language model that takes a whole-slide image (WSI) of breast cancer tissue and produces a structured pathology report **where every clinical sentence is spatially grounded** to the slide region it originates from.

Built on [CONCHv1.5](https://huggingface.co/MahmoodLab/conchv1_5) (frozen patch encoder) + [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) + LoRA, with a grounding loss that supervises text-to-vision attention during fine-tuning against BCSS semantic segmentation masks and CONCH v1 pseudo-GT.

## Why grounding matters

Existing pathology VLMs (PathChat, SlideChat, Quilt-LLaVA) produce plausible captions but cannot guarantee "this clinical claim came from this slide region." Grounding-aware approaches exist in radiology (MedGround 2026) but WSI-scale histopathology grounding is an open problem. PathoLens-VLM closes that gap.

---

## Headline Results

From the full-scale run `grounded_fullscale_20260604_021323` — 50 TCGA-BRCA slides (35 train / 8 val / 7 test), `configs/fullscale.yaml`:

| Metric | Value | Baseline |
|---|---|---|
| BLEU-4 (test, 7 slides) | **0.0559** | — |
| ROUGE-L (test, 7 slides) | **0.1990** | — |
| PG@5 (BCSS pointing game) | **0.081** | uniform: 0.019 (**4.4× lift**) |
| PG@10 | **0.081** | uniform: 0.037 (2.2×) |
| Top-10% attention mass | **0.201** | uniform: 0.100 (2.0×) |
| Entropy ratio | **0.938** | uniform: 1.000 (lower = peakier) |

> Full run report: [`docs/FULLSCALE_RUN_REPORT.md`](docs/FULLSCALE_RUN_REPORT.md)
> Smoke run history (bug audit + 3-run campaign): [`docs/SMOKE_RUN_REPORT.md`](docs/SMOKE_RUN_REPORT.md)

---

## Submission Deliverables

| Artifact | Path | Notes |
|---|---|---|
| One-page project HTML | [`index.html`](index.html) | Static page with title, team, description, tech stack, results, video placeholder |
| Project report (Word) | [`report/PathoLens_Report.docx`](report/PathoLens_Report.docx) | Filled-in ITU AI&DE template; rebuildable via `report/build_report.py` |
| Academic poster (A0) | [`poster/poster.html`](poster/poster.html) | A0 portrait, print to PDF from browser |

---

## Quick Start

### Requirements

- Python 3.11+, uv package manager
- PyTorch 2.5+ (CPU for local dev; CUDA 12.4 on RunPod)
- HuggingFace account with access to three gated models (see below)

### Install

```bash
git clone <repo-url>
cd patholens
make install         # uv venv + dependencies + pre-commit hooks
cp .env.example .env
# Edit .env: HF_TOKEN=hf_..., WANDB_API_KEY=..., RUNPOD_HOST=...
```

### Smoke tests (no GPU, no data required)

```bash
make test
```

Runs `tests/test_smoke.py`, `tests/test_grounding_targets.py`, and `tests/test_train_smoke.py`. All tests pass without HuggingFace dependencies or GPU — the LLM is replaced by a mock in the train smoke tests.

### HuggingFace gated model access (one-time)

Three models require institutional HF access approval. See `docs/MODEL_ACCESS.md` for the request links and expected approval time (~1–3 business days):

- `MahmoodLab/conchv1_5` (CONCHv1.5 patch encoder)
- `MahmoodLab/TITAN` (TITAN slide encoder — precomputed and stored but not in training loop)
- `meta-llama/Llama-3.2-3B-Instruct`

---

## Architecture

```
WSI (.svs, ~50K × 50K pixels)
  └─ Tissue segmentation (Otsu @1.25× / GrandQC fallback)
     └─ Patch extraction (448×448 @20×, ~3K–7K patches/slide)
        └─ CONCHv1.5 ViT-L  [FROZEN, 768-dim]
           │  patch_embeddings_v15: (N_patches, 768)
           │
           │  ← These are the LLM vision tokens (1:1 with grounding GT)
           ▼
        Linear Adapter  [TRAINABLE, 768→3072, ~5M params]
           ▼
        Llama-3.2-3B-Instruct + LoRA r=16  [TRAINABLE, ~10M params]
           │
           │  Input: [adapter_tokens (N_v)] + [instruction+response (T)]
           │  (LLaVA-style prepend, causal self-attention over all tokens)
           │
           ├─▶ Generated structured report
           └─▶ text_to_vision_attn (B, T, N_v)  ← grounding signal
                 extracted from Llama layer 14, averaged over all heads
```

**Key architectural decision:** CONCHv1.5 patch embeddings (not TITAN slide tokens) are used directly as LLM vision tokens. This ensures 1:1 correspondence between LLM attention weights and patch-level grounding GT. TITAN is precomputed and stored in HDF5 for potential future use.

**Trainable:** adapter (~5M) + LoRA (~10M) = ~15M total. Frozen: CONCHv1.5 (304M), Llama base.

See `docs/ARCHITECTURE.md` for full design rationale and all design decisions.

---

## Training

### Five Ablation Configs

| Config file | λ_g | Grounding source | Loss type |
|---|---|---|---|
| `caption_baseline.yaml` | 0 | — | — |
| `grounding_v1.yaml` | 0.3 | CONCH v1 pseudo-GT | KL divergence |
| `grounding_v1_cosine.yaml` | 0.3 | CONCH v1 pseudo-GT | cosine distance |
| `grounding_v2.yaml` | 0.3 | BCSS + CONCH hybrid | KL divergence |
| `grounding_v2_cosine.yaml` | 0.3 | BCSS + CONCH hybrid | cosine distance |

**v1 configs** use CONCH v1 zero-shot pseudo-GT (no additional data required beyond embeddings). **v2 configs** add BCSS patch-level segmentation masks where available (151 TCGA-BRCA slides, CC0 license).

### Loss Composition

```
L_total = L_caption + λ_g · L_grounding + λ_f · L_faithfulness

L_caption     = cross-entropy on response tokens (instruction masked with -100)
L_grounding   = KL(log_softmax(attn), gt_dist)   OR   1 - cosine_sim(attn, gt_dist)
L_faithfulness = mean entropy of per-sentence attention distributions
```

### Training Commands

```bash
# On RunPod:
make precompute                                        # CONCHv1.5 + CONCH v1 + TITAN embeddings (~10h)
make bcss-preprocess                                   # BCSS patch masks (~3h)

uv run python -m patholens.training.train \
    --config configs/grounding_v2.yaml \
    --run-name grounded_v2_run1                       # ~15h per run

make eval                                              # full evaluation suite
```

Or via the `make` targets:
```bash
make train-baseline
make train-grounded   # uses grounding_v2.yaml by default
```

### Optimizer

AdamW with separate learning rates:
- Linear adapter: `lr = 1e-4`
- LoRA parameters: `lr = 5e-5`

---

## Data

| Dataset | Role | Slides |
|---|---|---|
| TCGA-BRCA (BCSS-overlap) | Training + in-domain eval | 151 |
| BCSS annotations | Per-class grounding GT | 151 (CC0) |
| SlideInstruction | Instruction captions (GPT-4 structured) | 151 (filtered) |
| CAMELYON16 | Pointing-game evaluation only | ~100 |

All 151 training slides are TCGA-BRCA cases for which BCSS pixel-level ROI annotations exist (same SVS files, exact overlap). See `docs/DATA.md` for download instructions, preprocessing pipeline, and HDF5 schema.

---

## Evaluation

```bash
make eval   # runs all four eval modules
```

| Metric | Module | Ground Truth |
|---|---|---|
| BLEU-4, ROUGE-L | `evaluation/caption_metrics.py` | SlideInstruction captions |
| Attention concentration, entropy ratio | `evaluation/concentration_metric.py` | — (unsupervised) |
| Pointing Game @K (BCSS) | `evaluation/bcss_pointing_game.py` | BCSS pixel masks ✅ used in fullscale |
| Pointing Game @K (CAMELYON16) | `evaluation/pointing_game.py` | CAMELYON16 tumor masks _(not run — 150 GB dataset)_ |
| Faithfulness (intervention drop) | `evaluation/intervention_test.py` | Top-K patch masking _(planned)_ |
| Concept recall (tumor/stroma/lymph/necrosis) | `evaluation/concept_f1.py` | BCSS concept labels _(planned)_ |

Results are logged to `docs/EXPERIMENTS.md` via `/log-experiment`.

---

## Project Structure

```
patholens/
├── src/patholens/
│   ├── data/
│   │   ├── wsi_preprocessing.py      # Otsu/GrandQC tissue seg, patch extraction
│   │   ├── precompute_embeddings.py  # CONCHv1.5 + CONCH v1 + TITAN → HDF5
│   │   ├── bcss_loader.py            # BCSS download, rasterize, → HDF5 masks
│   │   ├── camelyon_xml_to_mask.py   # CAMELYON16 XML → patch-level binary mask
│   │   ├── conch_pseudo_gt.py        # PseudoGTCache: disk-backed (slide, sentence) → dist
│   │   └── dataset.py                # GroundingSlideDataset, grounding_collate_fn
│   ├── models/
│   │   ├── adapter.py                # LinearAdapter (768 → 3072)
│   │   ├── conch_encoder.py          # CONCHv1.5 + CONCH v1 loaders
│   │   ├── titan_encoder.py          # TITAN slide encoder wrapper
│   │   └── grounded_vlm.py           # GroundedVLM: adapter + Llama + LoRA
│   ├── training/
│   │   ├── losses.py                 # CombinedLoss (caption + grounding + faithfulness)
│   │   ├── grounding_targets.py      # build_target(): BCSS/pseudo-GT router
│   │   └── train.py                  # PathoTrainer (HF Trainer subclass), CLI entry point
│   ├── evaluation/
│   │   ├── pointing_game.py          # PG@K on CAMELYON16
│   │   ├── intervention_test.py      # Faithfulness via top-K patch masking
│   │   ├── caption_metrics.py        # BLEU/ROUGE
│   │   └── concept_f1.py             # Per-concept precision/recall/F1
│   └── utils/
│       ├── config.py                 # OmegaConf loader
│       └── logging.py                # setup_logger
├── tests/
│   ├── test_smoke.py                 # Import, adapter, loss, config tests
│   ├── test_grounding_targets.py     # BCSS routing and pseudo-GT fallback logic
│   └── test_train_smoke.py           # GroundedVLM + training E2E (MockLLM, no GPU)
├── configs/
│   ├── caption_baseline.yaml
│   ├── grounding_v1.yaml             # CONCH pseudo-GT, KL loss
│   ├── grounding_v1_cosine.yaml      # CONCH pseudo-GT, cosine loss
│   ├── grounding_v2.yaml             # BCSS hybrid, KL loss  ← recommended
│   └── grounding_v2_cosine.yaml      # BCSS hybrid, cosine loss
├── docs/
│   ├── ARCHITECTURE.md               # Design decisions, loss formulation, risks
│   ├── DATA.md                       # Download instructions, HDF5 schema, splits
│   ├── ROADMAP.md                    # 38-day week-by-week plan
│   ├── EXPERIMENTS.md                # Training run results (auto-updated)
│   ├── REMOTE_WORKFLOW.md            # RunPod setup and sync procedures
│   └── MODEL_ACCESS.md               # HuggingFace gated model access
├── scripts/
│   ├── 01_download_tcga.sh
│   ├── 02_download_camelyon.sh
│   └── 04_prepare_slideinstruction.py
└── Makefile
```

---

## Reproducibility Checklist

To reproduce the main `grounding_v2` result from scratch:

- [ ] `make install`
- [ ] Set `HF_TOKEN`, `WANDB_API_KEY` in `.env`
- [ ] Get HF access to CONCHv1.5, TITAN, Llama-3.2-3B-Instruct
- [ ] `make download-tcga` (RunPod, ~24h)
- [ ] `make bcss-preprocess` (~3h)
- [ ] `make precompute` (~10h)
- [ ] `make prepare-instruction`
- [ ] `uv run python -m patholens.training.train --config configs/grounding_v2.yaml --run-name grounded_v2_run1` (~15h)
- [ ] `make eval`
- [ ] Results in `docs/EXPERIMENTS.md`

---

## Hardware Budget

- **Local dev:** macOS, 20 GB AMD GPU + 32 GB RAM — all code development, smoke tests, notebook analysis
- **Cloud training:** RunPod A6000 48 GB @ $0.44/h
  - Embedding precompute: ~10h = ~$4.40
  - 5 training runs × ~15h = ~$33
  - Evaluation: ~5h = ~$2.20
  - **Total GPU: ~$40**
  - Storage (200 GB × 1.5 months × $0.07/GB): ~$21
  - **Safe upper bound: $80** (delete WSIs after precompute → ~$47)

---

## License

Research use only. Component licenses:

- CONCHv1.5: CC-BY-NC-ND 4.0 (non-commercial)
- TITAN: CC-BY-NC-ND 4.0
- BCSS annotations: CC0 (public domain)
- Llama-3.2: Meta Llama 3.2 Community License
- This repository: MIT

## Contact

ozf20@itu.edu.tr · ekere23@itu.edu.tr
