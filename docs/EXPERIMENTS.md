# Experiments Log

Manually maintained; newest run first. For the full methodology, audit history, and interpretation of these numbers, see [`SMOKE_RUN_REPORT.md`](./SMOKE_RUN_REPORT.md).

---

## 2026-06-02 16:14 — grounded_smoke_v3_20260602_161431

- **Config:** `configs/grounding_smoke_v2.yaml`
- **Hardware:** RunPod RTX A6000 48 GB
- **Training time:** 1 m 21 s (8 epochs × 8 train slides)
- **Data:** 8 train / 2 val / 10 test (BCSS ∩ SlideChat ∩ TCGA-BRCA-DX)
- **Hyperparameters:** lr_adapter=1e-4, lr_lora=5e-5, λ_grounding=0.1, λ_faithfulness=0.01, grounding_warmup_epochs=2, max_vision_tokens=1024, batch_size=1, grad_accum=2
- **Final losses (train, last step):** caption=2.18, grounding=0.74, faithfulness=6.92, total=2.32
- **eval_loss (val, epoch 8):** **2.32** ← lowest of the three smoke runs
- **Test metrics:**
  - **Caption:** BLEU-4 = **0.0566**, ROUGE-L = **0.1808** (10 slides)
  - **Grounding (concentration proxy):** top-10 % mass = **0.124**, entropy ratio = **0.999** (1.0 = uniform)
  - n_sentences_total = 22
- **Notes:**
  - First run after the four-bug audit (faithfulness softmax-of-prob, grounding double-softmax, dtype/device, vision-label padding).
  - First run with a real (non-no-op) grounding loss: dropped 5× from 7.7 (warmup end) to 0.5–0.7 (final).
  - Caption quality matches v1 ("lucky" no-op run); confirms grounding does not have to come at caption's expense if weights are tuned.
  - Faithfulness still locked at ~log(N_v); needs more data + slightly higher λ_f to bend the attention distribution.
- **W&B run:** `emirardaorigins-istanbul-technical-university/huggingface/runs/...` (see `wandb/run-20260602_161153-sh3ko0ty/`)
- **Checkpoint:** `checkpoints/grounded_smoke_v3_20260602_161431/final`
- **Result files:** `results/grounded_smoke_v3_20260602_161431_caption.json`, `..._grounding.json`

---

## 2026-06-02 12:13 — grounded_smoke_v2_20260602_121326

- **Config:** `configs/grounding_smoke.yaml` (unchanged from v1; reran *after* the loss bug-fixes were applied)
- **Hardware:** RunPod RTX A6000 48 GB
- **Training time:** ~1 m (4 epochs × 8 train slides)
- **Hyperparameters:** λ_grounding=0.3, λ_faithfulness=0.1, no warmup, epochs=4
- **Final losses:** caption≈5.95, grounding ∈ [2, 35] (spiky), faithfulness=6.9
- **eval_loss:** 4.37
- **Test metrics:**
  - **Caption:** BLEU-4 = **0.0055**, ROUGE-L = **0.0263** (10 slides)
  - Grounding concentration: top-10 % mass = 0.161, entropy_ratio = 0.988
- **Notes:**
  - Exposed the magnitude of the previously-suppressed grounding loss. With λ_g=0.3 and grounding values of 2–35, grounding dominates total loss → optimiser steals capacity from caption → BLEU collapses 10×.
  - **Diagnostic-only run.** Used to validate that the audit fixes were applied and to motivate the v3 retune.
- **Checkpoint:** `checkpoints/grounded_smoke_v2_20260602_121326/final`
- **Result files:** `results/grounded_smoke_v2_20260602_121326_caption.json`, `..._grounding.json`

---

## 2026-06-02 10:48 — grounded_smoke_20260602_105250

- **Config:** `configs/grounding_smoke.yaml`
- **Hardware:** RunPod RTX A6000 48 GB
- **Training time:** ~1 m (4 epochs × 8 train slides)
- **Hyperparameters:** λ_grounding=0.3, λ_faithfulness=0.1, no warmup, epochs=4
- **Final losses:** caption=1.86, grounding≈1e-8, faithfulness=6.93 (constant)
- **eval_loss:** 2.55
- **Test metrics:**
  - **Caption:** BLEU-4 = **0.0566**, ROUGE-L = **0.1916** (10 slides)
  - Grounding concentration: not measured (concentration metric written after this run)
- **Notes:**
  - The original smoke run. Numbers later proved **misleading**: grounding loss was a no-op because of a double-softmax bug; faithfulness was locked at log(N_v) because of a softmax-of-near-zero bug. The whole optimiser was effectively running caption-only, hence the relatively healthy BLEU.
  - Retained as historical baseline. Do **not** cite this BLEU as evidence of grounded training quality.
- **Checkpoint:** `checkpoints/grounded_smoke_20260602_105250/final` (used by the live Streamlit demo until the v3 checkpoint is wired in)
- **Result files:** `results/grounded_smoke_20260602_105250_caption.json`

---

## Run-to-run comparison

| Metric | v1 (broken losses) | v2 (bugs fixed, untuned) | **v3 (retuned + warmup)** |
|---|---|---|---|
| caption loss (final) | 1.86 | 5.95 | **1.4 – 2.2** ✅ |
| grounding loss (final) | 1e-8 (fake) | 2 – 35 (spiky) | **0.5 – 0.7** ✅ |
| faithfulness loss (final) | 6.93 const | 6.8 | 6.92 |
| eval_loss (val) | 2.55 | 4.37 | **2.32** ✅ |
| BLEU-4 | 0.0566 | 0.0055 | **0.0566** |
| ROUGE-L | 0.1916 | 0.0263 | **0.1808** |
| top-10 % attn mass | — | 0.161 | 0.124 |
| entropy ratio | — | 0.988 | 0.999 |

v3 is the only run that produces a real (non-no-op) grounding signal while keeping caption quality at the honest v1 level. It is the recommended baseline for the capstone write-up.
