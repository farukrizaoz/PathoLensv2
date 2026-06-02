"""Cheap grounding-quality proxy: attention concentration on generated sentences.

For each generated sentence we already have a (N_v,) attention distribution
(produced by `patholens.app.inference.generate_with_attention`). A model that
genuinely *grounds* should put most of its mass on a small fraction of patches;
a model that ignores vision should be near-uniform.

We report two scalars per sentence and aggregate at the corpus level:

    top-k mass   — sum of the top-k% largest attention values (higher = peakier)
    entropy      — Shannon entropy of the distribution in nats
                   (uniform over N_v patches → log(N_v); peaked on one patch → 0)

This is NOT the full pointing-game test (which needs CAMELYON16 pixel masks),
but it is a meaningful sanity check that runs in seconds on top of the existing
generator, so we can ship a grounding number for the report before the full
CAMELYON evaluation lands.

Usage:
    uv run python -m patholens.evaluation.concentration_metric \\
        --checkpoint checkpoints/<run>/final \\
        --test-set data/processed/slideinstruction_smoke/test.json \\
        --embeddings-dir data/processed/embeddings/tcga_brca \\
        --output-json results/<run>_grounding.json
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
import typer

from patholens.app.inference import DEFAULT_INSTRUCTION, generate_with_attention
from patholens.models.grounded_vlm import GroundedVLM
from patholens.utils.logging import setup_logger

app = typer.Typer()
logger = setup_logger(__name__)


def _load_vision(h5_path: Path, max_tokens: int, device: torch.device) -> torch.Tensor:
    with h5py.File(h5_path, "r") as f:
        emb = f["patch_embeddings_v15"][:]
    if emb.shape[0] > max_tokens:
        idx = np.linspace(0, emb.shape[0] - 1, max_tokens).astype(np.int64)
        emb = emb[idx]
    return torch.from_numpy(emb).to(device=device, dtype=torch.bfloat16).unsqueeze(0)


def _summarise(attn: np.ndarray, ks=(1, 5, 10)) -> dict:
    n = attn.size
    out: dict = {"n_vision_tokens": int(n)}
    sorted_a = np.sort(attn)[::-1]
    for k in ks:
        kk = max(1, int(round(n * k / 100)))
        out[f"top_{k}pct_mass"] = float(sorted_a[:kk].sum())
    eps = 1e-12
    p = attn / max(float(attn.sum()), eps)
    out["entropy_nats"] = float(-np.sum(p * np.log(p + eps)))
    out["entropy_uniform"] = float(np.log(n))
    out["entropy_ratio"] = out["entropy_nats"] / max(out["entropy_uniform"], eps)
    return out


@app.command()
def main(
    checkpoint: str = typer.Option(...),
    test_set: str = typer.Option(...),
    embeddings_dir: str = typer.Option("data/processed/embeddings/tcga_brca"),
    output_json: str = typer.Option("results/concentration.json"),
    max_vision_tokens: int = typer.Option(1024),
    max_new_tokens: int = typer.Option(384),
    instruction: str = typer.Option(DEFAULT_INSTRUCTION),
    limit: int = typer.Option(0),
    device: str = typer.Option("cuda"),
) -> None:
    """Generate reports + compute per-sentence attention concentration."""
    examples = json.load(open(test_set))
    if limit > 0:
        examples = examples[:limit]
    emb_dir = Path(embeddings_dir)
    examples = [e for e in examples if (emb_dir / f"{e['slide_id']}.h5").exists()]
    if not examples:
        raise SystemExit(f"No embeddings found for examples under {emb_dir}")
    logger.info("Evaluating concentration on %d slides", len(examples))

    model = GroundedVLM.from_pretrained(checkpoint).to(device)
    model.eval()
    dev = next(model.adapter.parameters()).device

    per_slide: list[dict] = []
    flat_summaries: list[dict] = []

    for i, ex in enumerate(examples):
        sid = ex["slide_id"]
        vision = _load_vision(emb_dir / f"{sid}.h5", max_vision_tokens, dev)
        out = generate_with_attention(
            model, vision_tokens=vision, instruction=instruction, max_new_tokens=max_new_tokens,
        )
        slide_summaries = [_summarise(a) for a in out["per_sentence_attn"]]
        flat_summaries.extend(slide_summaries)
        per_slide.append({
            "slide_id": sid,
            "n_sentences": len(out["sentences"]),
            "mean_top_10pct_mass": float(np.mean([s["top_10pct_mass"] for s in slide_summaries])) if slide_summaries else 0.0,
            "mean_entropy_ratio": float(np.mean([s["entropy_ratio"] for s in slide_summaries])) if slide_summaries else 0.0,
            "per_sentence": slide_summaries,
        })
        logger.info(
            "[%d/%d] %s: %d sent  top10%%=%.3f  entropy_ratio=%.3f",
            i + 1, len(examples), sid, len(out["sentences"]),
            per_slide[-1]["mean_top_10pct_mass"], per_slide[-1]["mean_entropy_ratio"],
        )

    corpus = {
        "n_slides": len(per_slide),
        "n_sentences_total": sum(p["n_sentences"] for p in per_slide),
        "corpus_top_1pct_mass": float(np.mean([s["top_1pct_mass"] for s in flat_summaries])) if flat_summaries else 0.0,
        "corpus_top_5pct_mass": float(np.mean([s["top_5pct_mass"] for s in flat_summaries])) if flat_summaries else 0.0,
        "corpus_top_10pct_mass": float(np.mean([s["top_10pct_mass"] for s in flat_summaries])) if flat_summaries else 0.0,
        "corpus_entropy_ratio": float(np.mean([s["entropy_ratio"] for s in flat_summaries])) if flat_summaries else 1.0,
    }
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"corpus": corpus, "per_slide": per_slide, "checkpoint": checkpoint, "test_set": test_set},
        open(out_path, "w"),
        indent=2,
    )
    logger.info(
        "Wrote %s — top-10%%=%.3f  entropy_ratio=%.3f  (1.0 = uniform, lower = more grounded)",
        out_path, corpus["corpus_top_10pct_mass"], corpus["corpus_entropy_ratio"],
    )


if __name__ == "__main__":
    app()
