# PathoLens-VLM

> Spatially-grounded vision-language model for whole-slide breast cancer histopathology report generation.
> ITU AI & Data Engineering Capstone, May–June 2026 (38-day window).
> Faruk Rıza Öz · Emir Arda Eker · Supervisor: Prof. Dr. Behçet Uğur Töreyin

## Research Question

A trainable VLM that explicitly guarantees which slide region each clinical sentence originates from, measurable via a faithfulness metric.

## Core Contribution

Adding a **grounding-aware report generation head** on top of TITAN (MahmoodLab, open) slide encoder + **Llama-3.2-3B + LoRA** to achieve **per-sentence spatial grounding** in histopathology reporting. Gap in existing work: TITAN is pretrained for retrieval/classification but not optimized for report generation; SlideChat generates captions but provides no grounding guarantee; MedGround addresses grounding in radiology but not histopathology.

## Architecture (One-Pass Summary)

```
WSI (.svs)
  └─ Tissue seg (Otsu / GrandQC fallback)
     └─ Patch extraction (448×448 @20×, ~3K-7K patches/slide)
        └─ CONCHv1.5 patch encoder (frozen)
           └─ TITAN slide encoder (frozen) → slide tokens [N_tokens × 768]
              └─ Linear adapter (TRAINABLE) → Llama hidden dim [3072]
                 └─ Llama-3.2-3B-Instruct + LoRA r=16 (TRAINABLE)
                    ├─ Generated structured report
                    ├─ Self-attention extraction (text→vision tokens)
                    │  → Per-sentence grounding heatmap
                    └─ → FHIR DiagnosticReport JSON
```

**Trainable:** adapter (~5M params) + LoRA (~10M params) = ~15M total.
**Frozen:** CONCHv1.5 (304M), TITAN slide encoder, Llama-3.2-3B base.

Details: `docs/ARCHITECTURE.md`.

## Data Summary

| Source | Size | Purpose |
|--------|------|---------|
| TCGA-BRCA | 150 slides + reports | Primary fine-tuning |
| CAMELYON16 (test) | ~100 slides + pixel masks | Pointing game evaluation |
| SlideInstruction | 4.2K WSI-caption + 176K VQA | Instruction tuning (open, GPT-4 structured) |

Details: `docs/DATA.md`.

## Stack

- Python 3.11+, PyTorch 2.5+, CUDA 12.4 (RunPod), uv package manager
- HuggingFace: `transformers`, `peft`, `accelerate`, `bitsandbytes` (RunPod-only)
- Models: `MahmoodLab/conchv1_5`, `MahmoodLab/TITAN`, `meta-llama/Llama-3.2-3B-Instruct` (all gated, institutional HF account required)
- WSI: `openslide-python`, `opencv-python-headless`, `h5py`
- MLOps: `wandb`
- Evaluation: BLEU/ROUGE (HF `evaluate`), custom pointing game

## Environment (Local vs RunPod)

**Local (macOS, 20 GB AMD GPU + 32 GB RAM):**
- All code development, testing, quick analysis, notebooks
- Local HDF5 embedding inspection and visualization
- Llama-3.2-3B 4-bit inference testing (CPU or small samples)

**Cloud (RunPod, A6000 48 GB ~$0.44/hr):**
- Embedding precompute (CONCHv1.5 + TITAN forward pass)
- LoRA fine-tuning
- Evaluation runs

**Total budget:**
- GPU: ~70–100h × $0.44 = $31–44
- Storage: 200 GB × 1.5 ay × $0.07/GB = $21 (WSI'lar embedding sonrası silinirse $6)
- **Gerçekçi toplam: $60–65 | Güvenli üst sınır: $80**
- Tasarruf: embedding bittikten sonra WSI'ları sil → ~$47'ye düşer

## Key Commands (all via `make`)

```bash
make install              # Install dependencies with uv + pre-commit
make test                 # Smoke tests (no GPU, no data required)
make lint && make format  # Ruff linting and formatting
make sync-up              # rsync local → RunPod
make sync-down            # Checkpoints + logs RunPod → local
make precompute           # CONCHv1.5 + TITAN embedding (RunPod)
make train-baseline       # Caption-only baseline training
make train-grounded       # Training with grounding loss
make eval                 # Full evaluation suite
```

## Work Division

**Faruk (Vision):** WSI preprocessing, CONCHv1.5 + TITAN embedding pipeline, grounding loss design, attention extraction, CAMELYON16 evaluation, faithfulness metric.

**Emir (Language):** Llama-3.2 LoRA setup, instruction-following format, SlideInstruction parsing, FHIR DiagnosticReport templating, caption metrics framework.

## Rules for Claude Code Sessions

**Follow these in every session:**

1. **Do not touch large files.** `data/`, `checkpoints/`, `wandb/`, `*.h5`, `*.svs`, `*.safetensors` — listed in `.claudeignore`, never load into context.
2. **Secret protection.** `HF_TOKEN`, `WANDB_API_KEY` live in `.env`. `.env` is git-ignored. NEVER commit.
3. **No architectural changes** without asking first: backbone swap, LLM selection, loss formulation. `docs/ARCHITECTURE.md` is the source of truth.
4. **Local-cloud discipline.** Code is written locally, pushed via `make sync-up` to RunPod, training runs there. Do not start training locally.
5. **No training without tests.** New loss/dataset/model → add a minimal smoke test to `tests/test_smoke.py` first.
6. **Lint discipline.** Run `make lint` before every commit. Do not commit with Ruff errors.
7. **Automatic experiment logging.** After every training run, execute `/log-experiment` → updates `docs/EXPERIMENTS.md`.
8. **Token efficiency.** Read long files with specific line ranges via `view`. Do not read the entire repo. Use `Glob` or `Grep` to locate needed files.

## Slash Skills (Token-Optimized Agents)

Defined under `.claude/skills/<name>/SKILL.md`. Each skill is **single-purpose, narrowly scoped**, wrapping raw commands:

- `/setup-env` — Local or RunPod environment setup (uv, .env, HF login)
- `/precompute-embeddings` — Embedding precompute pipeline (on RunPod)
- `/train-baseline` — Start caption-only baseline training
- `/train-grounded` — Start grounding loss training
- `/eval-pointing-game` — CAMELYON16 pointing game evaluation
- `/sync-runpod` — Local ↔ RunPod sync (push/pull selectable)
- `/log-experiment` — Log latest run to `EXPERIMENTS.md`
- `/diagnose-run` — Debug OOM/CUDA/loss explosion issues

Each skill contains its own usage rules. Invoke the skill for details, do not browse docs.

## Documentation Index

| File | Content |
|------|---------|
| `docs/ARCHITECTURE.md` | Detailed architectural decisions, loss design, hyperparameter rationale |
| `docs/DATA.md` | TCGA + CAMELYON + SlideInstruction download and preprocessing |
| `docs/ROADMAP.md` | 38-day day-by-day plan with responsibilities |
| `docs/REMOTE_WORKFLOW.md` | RunPod setup, sync, debugging procedures |
| `docs/EXPERIMENTS.md` | All training run results (auto-updated) |
| `docs/MODEL_ACCESS.md` | HuggingFace gated model access instructions |

## References

- TITAN: https://huggingface.co/MahmoodLab/TITAN
- CONCHv1.5: https://huggingface.co/MahmoodLab/conchv1_5
- Llama-3.2-3B: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- TITAN paper: https://arxiv.org/abs/2411.19666
- SlideChat (reference, not used): https://arxiv.org/abs/2410.11761
- MedGround (grounding reference): https://arxiv.org/abs/2601.06847
- Direct Visual Grounding (KL Attention Loss, 2025): https://arxiv.org/abs/2511.12738
