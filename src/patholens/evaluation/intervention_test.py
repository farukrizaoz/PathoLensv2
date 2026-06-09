"""Faithfulness evaluation via top-K patch masking (intervention test).

For each test slide:
1. Generate a report with the full embedding set → BLEU_original
2. Zero out the top-K attended patches (replace with the zero vector)
3. Re-generate → BLEU_masked
4. Faithfulness score = BLEU_original − BLEU_masked

A high drop indicates the model genuinely relies on the attended patches for
caption quality — strong evidence that attention is semantically grounded rather
than dispersed. A near-zero drop means the model ignores the attention (bad).

Usage:
    uv run python -m patholens.evaluation.intervention_test \\
        --checkpoint checkpoints/grounded_fullscale_20260604_021323/final \\
        --test-set data/processed/slideinstruction_fullscale/test.json \\
        --embeddings-dir data/processed/embeddings/tcga_brca \\
        --output-json results/intervention_test.json \\
        --top-k 10
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

MAX_VISION_TOKENS = 1024


# ─── Metric helpers ───────────────────────────────────────────────────────────


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def _ngrams(toks: list[str], n: int) -> list[tuple]:
    return [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]


def _bleu4(hypothesis: str, reference: str) -> float:
    """Corpus BLEU-4 (single-sentence, brevity-penalised)."""
    hyp = _tokens(hypothesis)
    ref = _tokens(reference)
    if not hyp:
        return 0.0
    import math

    bp = min(1.0, math.exp(1.0 - len(ref) / len(hyp))) if len(hyp) < len(ref) else 1.0
    score = 1.0
    for n in range(1, 5):
        hyp_ng = _ngrams(hyp, n)
        ref_ng = _ngrams(ref, n)
        if not hyp_ng:
            return 0.0
        ref_counts: dict[tuple, int] = {}
        for gram in ref_ng:
            ref_counts[gram] = ref_counts.get(gram, 0) + 1
        matches = sum(min(hyp_ng.count(gram), ref_counts.get(gram, 0)) for gram in set(hyp_ng))
        precision = matches / len(hyp_ng)
        if precision == 0:
            return 0.0
        score *= precision
    return bp * (score**0.25)


# ─── Intervention logic ───────────────────────────────────────────────────────


def _load_embeddings(
    h5_path: Path, max_tokens: int, device: torch.device
) -> tuple[torch.Tensor, np.ndarray]:
    """Load patch embeddings; return (1, N_v, 768) tensor and kept indices."""
    with h5py.File(h5_path, "r") as f:
        emb = f["patch_embeddings_v15"][:]
    if emb.shape[0] > max_tokens:
        idx = np.linspace(0, emb.shape[0] - 1, max_tokens).astype(np.int64)
        emb = emb[idx]
    else:
        idx = np.arange(emb.shape[0])
    tensor = torch.from_numpy(emb).to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    return tensor, idx


def run_intervention(
    model: GroundedVLM,
    embeddings: torch.Tensor,
    reference: str,
    top_k: int = 10,
    max_new_tokens: int = 256,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Run one intervention step for a single slide.

    Args:
        model: loaded GroundedVLM in eval mode
        embeddings: (1, N_v, 768) patch embeddings
        reference: ground-truth caption string
        top_k: number of top-attended patches to zero out
        max_new_tokens: generation length cap
        device: torch device

    Returns:
        dict with bleu_original, bleu_masked, bleu_drop, top_k
    """
    if device is None:
        device = next(model.parameters()).device

    instruction = DEFAULT_INSTRUCTION

    # Pass 1: generate with full embeddings
    output_orig, attn_orig = generate_with_attention(
        model, embeddings, instruction, max_new_tokens=max_new_tokens
    )
    bleu_orig = _bleu4(output_orig, reference)

    # Aggregate per-sentence attention → per-patch score
    if attn_orig is not None and attn_orig.ndim >= 2:
        patch_scores = attn_orig.mean(dim=0)  # (N_v,)
    else:
        # Fallback: no attention available — skip intervention
        return {
            "bleu_original": bleu_orig,
            "bleu_masked": bleu_orig,
            "bleu_drop": 0.0,
            "top_k": top_k,
        }

    n_v = embeddings.shape[1]
    top_k_clamped = min(top_k, n_v)
    top_indices = patch_scores.topk(top_k_clamped).indices

    # Pass 2: zero out top-K patches and regenerate
    masked_emb = embeddings.clone()
    masked_emb[0, top_indices] = 0.0
    output_masked, _ = generate_with_attention(
        model, masked_emb, instruction, max_new_tokens=max_new_tokens
    )
    bleu_masked = _bleu4(output_masked, reference)

    return {
        "bleu_original": bleu_orig,
        "bleu_masked": bleu_masked,
        "bleu_drop": bleu_orig - bleu_masked,
        "top_k": top_k_clamped,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────


@app.command()
def main(
    checkpoint: str = typer.Option(..., help="Path to trained model checkpoint"),
    test_set: str = typer.Option(..., help="Path to test JSON (slideinstruction format)"),
    embeddings_dir: str = typer.Option(..., help="Directory with per-slide HDF5 embeddings"),
    output_json: str = typer.Option("results/intervention_test.json"),
    top_k: int = typer.Option(10, help="Patches to zero out"),
    max_new_tokens: int = typer.Option(256),
) -> None:
    """Faithfulness test: measure BLEU drop after zeroing top-K attended patches."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model from {checkpoint}")
    model = GroundedVLM.from_pretrained(checkpoint)
    model.eval().to(device)

    test_path = Path(test_set)
    with test_path.open() as f:
        slides = json.load(f)

    emb_dir = Path(embeddings_dir)
    results: list[dict] = []

    for entry in slides:
        slide_id = entry["slide_id"]
        reference = entry.get("caption", "")
        h5_path = emb_dir / f"{slide_id}.h5"
        if not h5_path.exists():
            logger.warning(f"Missing embeddings for {slide_id}, skipping")
            continue

        embeddings, _ = _load_embeddings(h5_path, MAX_VISION_TOKENS, device)
        metrics = run_intervention(
            model, embeddings, reference, top_k=top_k, max_new_tokens=max_new_tokens
        )
        results.append({"slide_id": slide_id, **metrics})
        logger.info(
            f"{slide_id}: BLEU drop = {metrics['bleu_drop']:.4f} "
            f"({metrics['bleu_original']:.4f} → {metrics['bleu_masked']:.4f})"
        )

    mean_drop = float(np.mean([r["bleu_drop"] for r in results])) if results else 0.0
    summary = {
        "mean_bleu_drop": mean_drop,
        "top_k": top_k,
        "n_slides": len(results),
        "per_slide": results,
    }
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Mean BLEU drop @K={top_k}: {mean_drop:.4f}  →  {out_path}")


if __name__ == "__main__":
    app()
