# Full-Scale Run Report — `grounded_fullscale_20260604_021323`

**Generated:** 2026-06-04 04:11 UTC
**Pool:** 50 BCSS-covered TCGA-BRCA DX slides (intersection of BCSS ∩ SlideChat ∩ GDC).
**Split:** 35 train / 8 val / 7 test (deterministic, seed=42).
**Hardware:** RunPod RTX A6000 48 GB.

## Headline results

- **BLEU-4 (test):** 0.0559
- **ROUGE-L (test):** 0.1990
- **Concentration (top-10% attention mass):** 0.201 (random = 0.10)
- **Entropy ratio:** 0.938 (1.0 = uniform)
- **PG@5 (BCSS pointing-game):** 0.081 vs uniform 0.019 — **4.4× lift over random**
- **PG@10:** 0.081 vs uniform 0.037 — 2.2× lift
- **PG@20:** 0.108 vs uniform 0.071 — 1.5× lift

## Comparison vs smoke v3 baseline

| Metric | smoke v3 (10 slides) | **fullscale (50 slides)** | Δ |
|---|---|---|---|
| BLEU-4 | 0.0566 | **0.0559** | -0.0007 |
| ROUGE-L | 0.1808 | **0.1990** | +0.0182 |
| top-10% mass | 0.124 | **0.201** | +0.077 |
| entropy ratio | 0.999 | **0.938** | -0.061 |


## Training trajectory (mean loss per epoch)

| Epoch | caption | grounding | faithfulness | total |
|---|---|---|---|---|
| 0 | 11.235 | — | — | 11.235 |
| 1 | 2.699 | — | — | 2.699 |
| 2 | 2.314 | 11.753 | 6.590 | 3.066 |
| 3 | 1.974 | 3.546 | 6.891 | 2.673 |
| 4 | 2.088 | 14.018 | 6.148 | 3.797 |
| 5 | 1.770 | 0.257 | 6.929 | 2.142 |
| 6 | 1.869 | 10.374 | 6.350 | 3.224 |
| 7 | 1.770 | 0.574 | 6.928 | 2.173 |

### Held-out eval_loss

_(no eval_loss entries logged)_

## Per-slide caption metrics

| Slide | BLEU-4 | ROUGE-L |
|---|---|---|
| `TCGA-A2-A0T0-01Z-00-DX1` | 0.020 | 0.195 |
| `TCGA-A2-A3XU-01Z-00-DX1` | 0.104 | 0.225 |
| `TCGA-A2-A3XX-01Z-00-DX1` | 0.092 | 0.229 |
| `TCGA-A7-A0DA-01Z-00-DX1` | 0.031 | 0.155 |
| `TCGA-A1-A0SP-01Z-00-DX1` | 0.082 | 0.244 |
| `TCGA-A2-A0ST-01Z-00-DX1` | 0.018 | 0.135 |
| `TCGA-AO-A128-01Z-00-DX1` | 0.044 | 0.210 |


### BCSS pointing-game

> **Note.** The original chain's pointing-game run reported 0/71 concept-bearing
> sentences. Root cause: BCSS HDF5 keys are prefixed (`mask_tumor`,
> `mask_stroma`, …) but the channel lookup used the bare names. Fixed in
> `evaluation/bcss_pointing_game.py` and rerun; results below are from that
> v2 run (`results/<run>_pointing_v2.json`).

Evaluated on 7 held-out test slides, **37 concept-bearing sentences**
(34 skipped — sentences with no detectable BCSS concept word).

| K | **PG@K** | Uniform baseline | Lift |
|---|---|---|---|
| 1 | 0.000 | 0.004 | tied with random |
| 5 | **0.081** | 0.019 | **4.4×** |
| 10 | **0.081** | 0.037 | **2.2×** |
| 20 | **0.108** | 0.071 | 1.5× |

**Reading.** At K=5 the model points into the correct BCSS region 4.4× more
often than uniform random attention — a measurable grounding signal. The
effect is concentrated on slide `TCGA-AO-A128` (3/12 hits at K=5,
4/12 at K=20); the other six slides contribute 0 hits, suggesting either
slide-specific grounding fidelity or that the keyword detector picked up
sentences where the concept word appears in passing rather than as the
sentence's actual subject. PG@K should grow with longer training and more
slides; this is the publishable baseline.

## Per-slide pointing-game hits (v2 run)

| Slide | n_sentences | n_concept | hits@1 | hits@5 | hits@10 | hits@20 |
|---|---|---|---|---|---|---|
| `TCGA-A2-A0T0-01Z-00-DX1` | 7 | 6 | 0 | 0 | 0 | 0 |
| `TCGA-A2-A3XU-01Z-00-DX1` | 12 | 4 | 0 | 0 | 0 | 0 |
| `TCGA-A2-A3XX-01Z-00-DX1` | 9 | 3 | 0 | 0 | 0 | 0 |
| `TCGA-A7-A0DA-01Z-00-DX1` | 12 | 6 | 0 | 0 | 0 | 0 |
| `TCGA-A1-A0SP-01Z-00-DX1` | 11 | 3 | 0 | 0 | 0 | 0 |
| `TCGA-A2-A0ST-01Z-00-DX1` | 8 | 3 | 0 | 0 | 0 | 0 |
| `TCGA-AO-A128-01Z-00-DX1` | 12 | 12 | 0 | **3** | **3** | **4** |

## Configuration (`configs/fullscale.yaml`)

```yaml
# Full-scale run — 50 BCSS-covered TCGA-BRCA DX slides (35 train / 8 val / 7 test).
# Carries v3's retuned weights forward with λ_f bumped (more data per epoch means
# the regularizer has the gradient budget to actually bend attention).

run_name_prefix: grounded_fullscale

embeddings_dir: data/processed/embeddings/tcga_brca
instruction_train: data/processed/slideinstruction_fullscale/train.json
instruction_val:   data/processed/slideinstruction_fullscale/val.json
max_vision_tokens: 1024

bcss_masks_dir: data/processed/bcss_patch_masks
pseudo_gt_cache_dir: data/processed/pseudo_gt_cache

llm_repo: meta-llama/Llama-3.2-3B-Instruct
vision_dim: 768
llm_hidden_dim: 3072
load_in_4bit: true

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  bias: none

loss:
  lambda_grounding: 0.1
  lambda_faithfulness: 0.05      # bumped from 0.01 — more data per epoch
  grounding_warmup_epochs: 2.0
  loss_type: kl

grounding:
  source: bcss_hybrid
  conch_repo: MahmoodLab/conchv1_5
  cosine_temperature: 0.1

attn_layer: 14

optimizer:
  name: adamw
  lr_adapter: 1.0e-4
  lr_lora: 5.0e-5
  weight_decay: 0.01

scheduler:
  name: cosine
  warmup_steps: 8

training:
  epochs: 8
  batch_size: 1
  gradient_accumulation: 4       # bumped from 2 — smoother gradients at scale
  mixed_precision: bf16
  gradient_checkpointing: true
  max_grad_norm: 1.0
  eval_every: 35                 # ~ once per epoch (35 train steps/epoch)
  save_every: 70                 # ~ every other epoch

output_dir: checkpoints

wandb:
  project: patholens-vlm
  tags: [fullscale, bcss-hybrid, kl, retuned, n50]
log_every: 4
```

## Reproduction

```bash
# 0. Restore environment on a fresh pod with /workspace volume attached
cd /workspace/patholens-vlm && uv sync && source .env

# 1. Run the full chain (resume-safe; rerun any time, skips completed phases)
bash scripts/fullscale_runner.sh

# 2. Artifacts
ls checkpoints/grounded_fullscale_20260604_021323/final/        # trained model
ls results/grounded_fullscale_20260604_021323_*.json            # caption / grounding / pointing metrics
ls /workspace/logs/fullscale/      # per-phase logs + flag files
```

## Notes

- BCSS pointing-game uses BCSS pixel masks as ground truth; eval slides are
  **disjoint from training slides** (deterministic 35/8/7 split), so PG@K is a
  generalisation test of the grounding mechanism, not memorisation.
- The 10 smoke-run slides are *included* in the fullscale training pool (they
  were already precomputed). The smoke-run checkpoint can still be loaded
  separately from `checkpoints/grounded_smoke_v3_*/final` for comparison.
- All four loss-correctness fixes from the smoke audit (faithfulness softmax,
  grounding double-softmax, adapter dtype/device, vision-label padding) remain
  in place; verified by `scripts/dry_run_pipeline.py` at phase 4 of the runner.
