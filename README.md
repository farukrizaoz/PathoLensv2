# PathoLens-VLM

**Spatially-Grounded Vision-Language Model for Whole-Slide Breast Cancer Histopathology Report Generation**

Istanbul Technical University — AI & Data Engineering Capstone Project (2026)

Faruk Rıza Öz · Emir Arda Eker · Supervisor: Prof. Dr. Behçet Uğur Töreyin

---

## What this is

A vision-language model that takes a whole-slide image (WSI) of breast cancer tissue and produces a structured pathology report **where every clinical sentence is spatially grounded** to the slide region it came from. Output: HL7 FHIR-compatible DiagnosticReport JSON + per-sentence attention heatmaps.

Built on top of [TITAN](https://huggingface.co/MahmoodLab/TITAN) (MahmoodLab's slide-level pathology foundation model) and [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), with a novel **grounding loss** that supervises text-to-vision attention during fine-tuning.

## Why grounding matters

Existing pathology VLMs (PathChat, SlideChat, Quilt-LLaVA) produce plausible captions and answer VQA but cannot guarantee that "this clinical claim came from this region of the slide." Grounding-aware approaches have appeared in radiology this year (MedGround Jan 2026, MedMO Feb 2026) but **WSI-scale histopathology grounding remains an open problem**. We close that gap.

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd patholens-vlm
make install-dev

# 2. Set up environment
cp .env.example .env
# Edit .env: HF_TOKEN, WANDB_API_KEY, RUNPOD_HOST

# 3. Smoke tests (no GPU, no data)
make test

# 4. Get HF gated access (one-time, see docs/MODEL_ACCESS.md)
#    - MahmoodLab/conchv1_5
#    - MahmoodLab/TITAN
#    - meta-llama/Llama-3.2-3B-Instruct

# 5. Local dev → RunPod training cycle
make sync-up                    # local code → RunPod
# On RunPod:
make precompute                 # one-time, ~10h
make train-baseline             # ~12h
make train-grounded             # ~18h
make eval                       # full eval suite
make sync-down                  # results → local
```

## Architecture

```
WSI (gigapixel, .svs)
  └─ Tissue segmentation (Otsu / GrandQC fallback)
     └─ Patch extraction (448×448 @20×, ~3K-7K patches/slide)
        └─ CONCHv1.5 patch encoder (frozen, 768-dim)
           └─ TITAN slide encoder (frozen, slide-level transformer)
              └─ Linear adapter (trainable)
                 └─ Llama-3.2-3B-Instruct + LoRA r=16 (trainable)
                    ├─ Generated structured report (text)
                    ├─ Self-attention extraction (text→vision tokens)
                    │  → Per-sentence grounding heatmap
                    └─ FHIR DiagnosticReport JSON
```

Frozen: ~330M parameters. Trainable: ~15M (adapter + LoRA).

See `docs/ARCHITECTURE.md` for design rationale.

## Datasets

| Dataset            | Use                               | Size                         |
| ------------------ | --------------------------------- | ---------------------------- |
| TCGA-BRCA (subset) | Main fine-tuning                  | 150 slides                   |
| CAMELYON16 (test)  | Pointing-game ground truth        | ~100 slides + pixel masks    |
| SlideInstruction   | Instruction-following data (open) | 4.2K WSI captions + 176K VQA |

See `docs/DATA.md` for download and preprocessing.

## Training

Three losses combined:

1. **Caption loss** — standard LM cross-entropy
2. **Grounding loss** — text-to-vision attention KL-divergence against CONCHv1.5 text-image cosine similarity (pseudo-supervision on TCGA, explicit supervision on CAMELYON)
3. **Faithfulness regularization** — attention concentration penalty (each sentence attends to focused subset of slide tokens)

Configs: `configs/caption_baseline.yaml`, `configs/grounding_v1.yaml`, `configs/grounding_v2.yaml`.

## Evaluation

- **Pointing game** (CAMELYON16): does "metastasis" sentence attention overlap with pixel-level tumor mask?
- **Intervention test**: mask top-K attention patches → does sentence change?
- **Caption metrics**: BLEU, ROUGE-L, METEOR vs ground-truth reports
- **Faithfulness score**: aligned attention concentration vs random baseline

## Project Structure

```
patholens-vlm/
├── CLAUDE.md                # Claude Code session context
├── README.md
├── pyproject.toml           # uv package config
├── Makefile                 # All critical commands
├── docs/                    # Architecture, data, roadmap, etc.
├── src/patholens/           # Python package
│   ├── data/                # WSI ingestion, datasets
│   ├── models/              # Adapter, grounded VLM, TITAN wrapper
│   ├── training/            # Losses, trainer, train entrypoint
│   ├── evaluation/          # Pointing game, intervention test
│   ├── reporting/           # FHIR DiagnosticReport templating
│   └── utils/               # Logging, config, IO
├── scripts/                 # Setup, download, sync
├── configs/                 # YAML training configs
├── notebooks/               # Exploration, visualization
├── tests/                   # pytest smoke tests
└── .claude/skills/          # Custom slash commands (token-optimized)
```

## Hardware

- **Local dev:** macOS, 20GB AMD GPU + 32GB RAM
- **Cloud train:** RunPod RTX A6000 48GB (~$0.44/h, total budget ~$35)

## Status

🚧 In active development (38-day capstone window: May-June 2026).
See `docs/ROADMAP.md` for week-by-week plan.

## License

Research use only. Built on multiple open-source models with their own licenses:

- CONCHv1.5: CC-BY-NC-ND 4.0 (non-commercial)
- TITAN: CC-BY-NC-ND 4.0
- Llama-3.2: Meta Llama 3.2 Community License

## Contact

ozf20@itu.edu.tr · ekere23@itu.edu.tr
