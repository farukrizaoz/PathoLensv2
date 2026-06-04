"""Generate docs/FULLSCALE_RUN_REPORT.md from training log + result JSONs.

Pulls:
  - configs/fullscale.yaml                — hyperparameters
  - /workspace/logs/fullscale/train.log    — per-step losses
  - results/<run>_caption.json             — BLEU/ROUGE
  - results/<run>_grounding.json           — concentration
  - results/<run>_pointing.json            — BCSS pointing game

Writes a single markdown file with: setup, training trajectory, all metrics,
side-by-side vs smoke v3, per-slide tables, and reproduction recipe.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from datetime import datetime, timezone

SMOKE_V3 = {
    "eval_loss": 2.32, "bleu4": 0.0566, "rouge_l": 0.1808,
    "top_10pct_mass": 0.124, "entropy_ratio": 0.999,
    "n_train": 8, "n_val": 2, "n_test": 10, "epochs": 8,
}


def _load_json(p: Path):
    return json.load(open(p)) if p.exists() else None


def _parse_train_log(log: Path) -> list[dict]:
    """Pull per-step loss dicts out of the train log."""
    if not log.exists():
        return []
    pat = re.compile(r"\{'(loss|train/caption)[^}]*\}")
    rows: list[dict] = []
    for line in log.read_text(errors="ignore").splitlines():
        for m in pat.finditer(line):
            try:
                # python repr from HF; eval is safe-ish on `loss=...` dicts but use ast
                import ast
                rows.append(ast.literal_eval(m.group(0)))
            except Exception:
                pass
    return rows


def _trajectory_table(rows: list[dict]) -> str:
    if not rows:
        return "_(no training log available)_\n"
    epochs = {}
    for r in rows:
        ep = r.get("epoch")
        if ep is None:
            continue
        ep_int = int(round(float(ep)))
        epochs.setdefault(ep_int, []).append(r)
    out = ["| Epoch | caption | grounding | faithfulness | total |", "|---|---|---|---|---|"]
    for ep in sorted(epochs):
        block = epochs[ep]
        cap = [r["train/caption"] for r in block if "train/caption" in r]
        gnd = [r["train/grounding"] for r in block if "train/grounding" in r]
        fth = [r["train/faithfulness"] for r in block if "train/faithfulness" in r]
        tot = [r["train/total"] for r in block if "train/total" in r]
        def avg(xs): return f"{sum(xs)/len(xs):.3f}" if xs else "—"
        out.append(f"| {ep} | {avg(cap)} | {avg(gnd)} | {avg(fth)} | {avg(tot)} |")
    return "\n".join(out)


def _eval_loss_trace(rows: list[dict]) -> str:
    eval_rows = [r for r in rows if "eval_loss" in r]
    if not eval_rows:
        return "_(no eval_loss entries logged)_"
    return "\n".join(
        f"- epoch {r.get('epoch','?')}: eval_loss = **{r['eval_loss']:.4f}**" for r in eval_rows
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--output", default="docs/FULLSCALE_RUN_REPORT.md")
    ap.add_argument("--log-dir", default="/workspace/logs/fullscale")
    ap.add_argument("--config", default="configs/fullscale.yaml")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    run = args.run_name
    res_dir = Path(args.results_dir)
    cap = _load_json(res_dir / f"{run}_caption.json") or {}
    gnd = _load_json(res_dir / f"{run}_grounding.json") or {}
    pnt = _load_json(res_dir / f"{run}_pointing.json") or {}
    train_rows = _parse_train_log(Path(args.log_dir) / "train.log")
    cfg = Path(args.config).read_text() if Path(args.config).exists() else "(missing)"

    cap_c = cap.get("corpus", {})
    gnd_c = gnd.get("corpus", {})
    pnt_c = pnt.get("corpus", {})

    bleu = cap_c.get("bleu4_mean", 0.0)
    rouge = cap_c.get("rouge_l_mean", 0.0)
    top10 = gnd_c.get("corpus_top_10pct_mass", 0.0)
    ent_r = gnd_c.get("corpus_entropy_ratio", 1.0)
    n_concept = pnt_c.get("n_concept_sentences", 0)

    # Build per-slide caption table
    cap_rows = cap.get("per_slide", [])
    cap_tbl = "| Slide | BLEU-4 | ROUGE-L |\n|---|---|---|\n"
    for r in cap_rows:
        cap_tbl += f"| `{r['slide_id']}` | {r.get('bleu4', 0):.3f} | {r.get('rouge_l', 0):.3f} |\n"

    # Per-slide pointing-game
    pnt_rows = pnt.get("per_slide", [])
    pnt_tbl_lines = ["| Slide | sentences | concept-sents | hits@1 | hits@5 | hits@10 |",
                     "|---|---|---|---|---|---|"]
    for r in pnt_rows:
        h = r.get("hits_by_k", {})
        pnt_tbl_lines.append(
            f"| `{r['slide_id']}` | {r['n_sentences']} | {r['n_concept_sentences']} | "
            f"{h.get('1', h.get(1, 0))} | {h.get('5', h.get(5, 0))} | {h.get('10', h.get(10, 0))} |"
        )
    pnt_tbl = "\n".join(pnt_tbl_lines)

    smoke_compare = (
        "| Metric | smoke v3 (10 slides) | **fullscale (50 slides)** | Δ |\n"
        "|---|---|---|---|\n"
        f"| BLEU-4 | {SMOKE_V3['bleu4']:.4f} | **{bleu:.4f}** | {bleu - SMOKE_V3['bleu4']:+.4f} |\n"
        f"| ROUGE-L | {SMOKE_V3['rouge_l']:.4f} | **{rouge:.4f}** | {rouge - SMOKE_V3['rouge_l']:+.4f} |\n"
        f"| top-10% mass | {SMOKE_V3['top_10pct_mass']:.3f} | **{top10:.3f}** | {top10 - SMOKE_V3['top_10pct_mass']:+.3f} |\n"
        f"| entropy ratio | {SMOKE_V3['entropy_ratio']:.3f} | **{ent_r:.3f}** | {ent_r - SMOKE_V3['entropy_ratio']:+.3f} |\n"
    )

    pointing_md = ""
    if pnt_c:
        pointing_md = (
            "### BCSS pointing-game\n\n"
            f"Evaluated on {pnt_c.get('n_slides', 0)} held-out test slides, "
            f"{pnt_c.get('n_concept_sentences', 0)} concept-bearing sentences "
            f"(skipped {pnt_c.get('n_skipped_no_concept', 0)} with no detectable concept).\n\n"
            "| K | PG@K | Uniform baseline | Δ |\n|---|---|---|---|\n"
        )
        for k in (pnt.get("top_ks") or [1, 5, 10, 20]):
            pg = pnt_c.get(f"PG@{k}", 0.0)
            u = pnt_c.get(f"PG_uniform@{k}", 0.0)
            pointing_md += f"| {k} | **{pg:.3f}** | {u:.3f} | {pg - u:+.3f} |\n"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    doc = f"""# Full-Scale Run Report — `{run}`

**Generated:** {now}
**Pool:** 50 BCSS-covered TCGA-BRCA DX slides (intersection of BCSS ∩ SlideChat ∩ GDC).
**Split:** 35 train / 8 val / 7 test (deterministic, seed=42).
**Hardware:** RunPod RTX A6000 48 GB.

## Headline results

- **BLEU-4 (test):** {bleu:.4f}
- **ROUGE-L (test):** {rouge:.4f}
- **Concentration (top-10% attention mass):** {top10:.3f} (random = 0.10)
- **Entropy ratio:** {ent_r:.3f} (1.0 = uniform)
{f"- **PG@5 (BCSS pointing-game):** {pnt_c.get('PG@5', 0.0):.3f} (uniform = {pnt_c.get('PG_uniform@5', 0.0):.3f})" if pnt_c else ""}

## Comparison vs smoke v3 baseline

{smoke_compare}

## Training trajectory (mean loss per epoch)

{_trajectory_table(train_rows)}

### Held-out eval_loss

{_eval_loss_trace(train_rows)}

## Per-slide caption metrics

{cap_tbl}

{pointing_md}

## Per-slide pointing-game hits

{pnt_tbl}

## Configuration (`{args.config}`)

```yaml
{cfg.strip()}
```

## Reproduction

```bash
# 0. Restore environment on a fresh pod with /workspace volume attached
cd /workspace/patholens-vlm && uv sync && source .env

# 1. Run the full chain (resume-safe; rerun any time, skips completed phases)
bash scripts/fullscale_runner.sh

# 2. Artifacts
ls checkpoints/{run}/final/        # trained model
ls results/{run}_*.json            # caption / grounding / pointing metrics
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
"""
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    print(f"Wrote {out}  ({len(doc):,} bytes)")


if __name__ == "__main__":
    main()
