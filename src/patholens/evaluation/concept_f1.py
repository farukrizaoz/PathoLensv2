"""Concept recall evaluation: does the generated report mention BCSS concepts?

For each test slide we know which BCSS concepts are present (tumor, stroma,
lymph, necrosis) by checking which `mask_*` channels in the slide's BCSS HDF5
have nonzero coverage.  We then check whether the generated caption mentions
keyword synonyms for each expected concept.

This is a keyword-based recall metric — not a clinical F1 — but it gives a
lightweight, reproducible signal for whether the model's vocabulary aligns with
the tissue content.  No TCGA clinical TSV required.

Reported per-concept:
    precision = TP / (TP + FP)   — when model mentions concept, was it present?
    recall    = TP / (TP + FN)   — when concept was present, did model mention it?
    F1        = harmonic mean

Usage:
    uv run python -m patholens.evaluation.concept_f1 \\
        --checkpoint checkpoints/grounded_fullscale_20260604_021323/final \\
        --test-set data/processed/slideinstruction_fullscale/test.json \\
        --embeddings-dir data/processed/embeddings/tcga_brca \\
        --bcss-dir data/processed/bcss_patch_masks \\
        --output-json results/concept_f1.json
"""

from __future__ import annotations

import json
import re
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
BCSS_COVERAGE_THRESHOLD = 0.05  # ≥5% of patches must be covered for concept to count

CONCEPT_KEYWORDS: dict[str, list[str]] = {
    "tumor": [
        "tumor",
        "tumour",
        "carcinoma",
        "invasive",
        "malignant",
        "cancer",
        "neoplasm",
        "neoplastic",
        "ductal",
        "lobular",
        "IDC",
        "ILC",
    ],
    "stroma": [
        "stroma",
        "stromal",
        "fibrosis",
        "fibrous",
        "desmoplastic",
        "connective tissue",
        "fibroblast",
    ],
    "lymph": [
        "lymphocyte",
        "lymphocytic",
        "inflammation",
        "inflammatory",
        "infiltrate",
        "infiltrating",
        "TIL",
        "TILs",
        "immune",
        "plasma cell",
        "lymphoid",
    ],
    "necrosis": [
        "necrosis",
        "necrotic",
        "debris",
        "dead cell",
        "karyorrhexis",
        "coagulative",
    ],
}


# ─── Core helpers ─────────────────────────────────────────────────────────────


def _concept_present_in_text(text: str, concept: str) -> bool:
    """Return True if any keyword for the concept appears in text (word boundary match)."""
    for kw in CONCEPT_KEYWORDS[concept]:
        if re.search(r"\b" + re.escape(kw.lower()) + r"\b", text.lower()):
            return True
    return False


def compute_concept_recall(
    generated: str,
    gt_concepts: list[str],
) -> dict[str, dict[str, int]]:
    """Check keyword presence for each concept against GT concept list.

    Args:
        generated: generated report text
        gt_concepts: list of concept names present for this slide
                     (e.g. ["tumor", "stroma"])

    Returns:
        per-concept dict with keys "tp", "fp", "fn"
    """
    results: dict[str, dict[str, int]] = {}
    for concept in CONCEPT_KEYWORDS:
        predicted = _concept_present_in_text(generated, concept)
        label = concept in gt_concepts
        results[concept] = {
            "tp": int(predicted and label),
            "fp": int(predicted and not label),
            "fn": int(not predicted and label),
        }
    return results


def _aggregate_f1(per_slide: list[dict[str, dict[str, int]]]) -> dict[str, dict[str, float]]:
    """Micro-aggregate TP/FP/FN across slides and compute P/R/F1 per concept."""
    totals: dict[str, dict[str, int]] = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CONCEPT_KEYWORDS}
    for slide in per_slide:
        for concept, counts in slide.items():
            for key, val in counts.items():
                totals[concept][key] += val

    out: dict[str, dict[str, float]] = {}
    for concept, counts in totals.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[concept] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out


def _get_gt_concepts(bcss_path: Path) -> list[str]:
    """Return list of BCSS concepts with nonzero coverage in this slide's mask file."""
    concepts = []
    try:
        with h5py.File(bcss_path, "r") as f:
            for concept in CONCEPT_KEYWORDS:
                key = f"mask_{concept}"
                if key in f:
                    coverage = np.asarray(f[key][:], dtype=np.float32)
                    if coverage.mean() >= BCSS_COVERAGE_THRESHOLD:
                        concepts.append(concept)
    except Exception as exc:
        logger.warning(f"Could not read BCSS file {bcss_path}: {exc}")
    return concepts


def _load_embeddings(h5_path: Path, max_tokens: int, device: torch.device) -> torch.Tensor:
    with h5py.File(h5_path, "r") as f:
        emb = f["patch_embeddings_v15"][:]
    if emb.shape[0] > max_tokens:
        idx = np.linspace(0, emb.shape[0] - 1, max_tokens).astype(np.int64)
        emb = emb[idx]
    return torch.from_numpy(emb).to(device=device, dtype=torch.bfloat16).unsqueeze(0)


# ─── CLI ──────────────────────────────────────────────────────────────────────


@app.command()
def main(
    checkpoint: str = typer.Option(..., help="Path to trained model checkpoint"),
    test_set: str = typer.Option(..., help="Path to test JSON (slideinstruction format)"),
    embeddings_dir: str = typer.Option(..., help="Directory with per-slide HDF5 embeddings"),
    bcss_dir: str = typer.Option(..., help="Directory with per-slide BCSS mask HDF5 files"),
    output_json: str = typer.Option("results/concept_f1.json"),
    max_new_tokens: int = typer.Option(256),
) -> None:
    """Concept recall evaluation: keyword-based P/R/F1 vs BCSS concept labels."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model from {checkpoint}")
    model = GroundedVLM.from_pretrained(checkpoint)
    model.eval().to(device)

    test_path = Path(test_set)
    with test_path.open() as f:
        slides = json.load(f)

    emb_dir = Path(embeddings_dir)
    bcss_root = Path(bcss_dir)
    per_slide_results: list[dict] = []

    for entry in slides:
        slide_id = entry["slide_id"]
        h5_path = emb_dir / f"{slide_id}.h5"
        bcss_path = bcss_root / f"{slide_id}.h5"

        if not h5_path.exists():
            logger.warning(f"Missing embeddings for {slide_id}, skipping")
            continue

        gt_concepts = _get_gt_concepts(bcss_path)
        embeddings = _load_embeddings(h5_path, MAX_VISION_TOKENS, device)
        generated, _ = generate_with_attention(
            model, embeddings, DEFAULT_INSTRUCTION, max_new_tokens=max_new_tokens
        )
        concept_scores = compute_concept_recall(generated, gt_concepts)
        per_slide_results.append(
            {"slide_id": slide_id, "gt_concepts": gt_concepts, "concepts": concept_scores}
        )
        logger.info(f"{slide_id}: gt={gt_concepts} scores={concept_scores}")

    aggregated = _aggregate_f1([r["concepts"] for r in per_slide_results])
    macro_f1 = float(np.mean([v["f1"] for v in aggregated.values()]))
    summary = {
        "macro_f1": round(macro_f1, 4),
        "per_concept": aggregated,
        "n_slides": len(per_slide_results),
        "per_slide": per_slide_results,
    }
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Macro-F1: {macro_f1:.4f}  →  {out_path}")


if __name__ == "__main__":
    app()
