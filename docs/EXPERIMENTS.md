# Experiments Log

Auto-updated by `/log-experiment` slash command after each training run.

Format: en yeni run başta.

---

## Şablon

```markdown
### YYYY-MM-DD HH:MM — run_name

- **Config:** `configs/xxx.yaml`
- **Hardware:** RunPod A6000 48GB (or local)
- **Training time:** Xh Ym
- **Data:** N slides
- **Hyperparameters:** lr=X, lambda_grounding=X, lambda_faithfulness=X
- **Metrics:**
  - Caption: BLEU-4=X, ROUGE-L=X, METEOR=X
  - Grounding: PG@5=X, PG@10=X
  - Faithfulness: intervention_drop=X
- **Notes:** ...
- **W&B run:** [link]
- **Checkpoint:** `checkpoints/xxx/`
```

---

## Henüz çalıştırılmış run yok.

İlk `make train-baseline` sonrası buraya işlenecek.
