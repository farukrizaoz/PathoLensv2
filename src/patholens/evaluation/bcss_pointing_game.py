"""BCSS-pointing game: did the model's per-sentence attention land inside
the human-annotated BCSS region for that sentence's concept?

For every generated sentence we have a (N_v,) attention distribution.
For every test slide we have a BCSS per-patch coverage matrix (TUMOR / STROMA /
LYMPH / NECROSIS / roi_coverage). When a generated sentence contains a concept
keyword, the model's top-K attended patches should overlap the patches BCSS
marks as that concept.

We report:
    PG@K  — fraction of concept-sentences for which any of top-K attended
            patches has BCSS coverage ≥ threshold for the matching concept
    PG_uniform — fraction expected by uniform random attention (sanity baseline)
    n_concept_sentences — how many sentences actually had a routable concept

This uses BCSS as ground truth on held-out *slides* not seen during training,
so it is a genuine generalization test of the grounding mechanism.
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

# Map concept keywords to BCSS channels (matches grounding_targets._detect_concept)
CONCEPT_TO_CHANNEL: dict[str, str] = {
    "tumor": "tumor", "tumour": "tumor", "carcinoma": "tumor", "invasive": "tumor",
    "neoplas": "tumor", "malignan": "tumor", "cancer": "tumor",
    "stroma": "stroma", "stromal": "stroma", "fibrous": "stroma",
    "lymph": "lymph", "infiltrate": "lymph", "tils": "lymph", "tilss": "lymph",
    "lymphocyt": "lymph", "plasma cell": "lymph",
    "necros": "necrosis", "necrotic": "necrosis",
}
BCSS_THRESHOLD = 0.20


def _load_vision(h5: Path, max_tok: int, device) -> tuple[torch.Tensor, np.ndarray]:
    with h5py.File(h5, "r") as f:
        emb = f["patch_embeddings_v15"][:]
    if emb.shape[0] > max_tok:
        idx = np.linspace(0, emb.shape[0] - 1, max_tok).astype(np.int64)
        emb = emb[idx]
        return torch.from_numpy(emb).to(device=device, dtype=torch.bfloat16).unsqueeze(0), idx
    return torch.from_numpy(emb).to(device=device, dtype=torch.bfloat16).unsqueeze(0), np.arange(emb.shape[0])


def _load_bcss(path: Path, kept_idx: np.ndarray) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        out = {k: f[k][:][kept_idx] for k in f.keys()}
    return out


def _detect_concept(sentence: str) -> str | None:
    s = sentence.lower()
    for kw, ch in CONCEPT_TO_CHANNEL.items():
        if kw in s:
            return ch
    return None


@app.command()
def main(
    checkpoint: str = typer.Option(...),
    test_set: str = typer.Option(...),
    embeddings_dir: str = typer.Option("data/processed/embeddings/tcga_brca"),
    bcss_masks_dir: str = typer.Option("data/processed/bcss_patch_masks"),
    output_json: str = typer.Option("results/pointing.json"),
    max_vision_tokens: int = typer.Option(1024),
    max_new_tokens: int = typer.Option(384),
    top_ks: list[int] = typer.Option([1, 5, 10, 20]),
    instruction: str = typer.Option(DEFAULT_INSTRUCTION),
    device: str = typer.Option("cuda"),
) -> None:
    examples = json.load(open(test_set))
    emb_dir = Path(embeddings_dir)
    bcss_dir = Path(bcss_masks_dir)
    examples = [e for e in examples if (emb_dir / f"{e['slide_id']}.h5").exists()]
    examples = [e for e in examples if (bcss_dir / f"{e['slide_id']}.h5").exists()]
    if not examples:
        raise SystemExit("No slides have both an embeddings H5 and a BCSS mask H5")
    logger.info("BCSS pointing-game on %d slides", len(examples))

    model = GroundedVLM.from_pretrained(checkpoint).to(device)
    model.eval()
    dev = next(model.adapter.parameters()).device

    per_slide: list[dict] = []
    hits_by_k: dict[int, list[int]] = {k: [] for k in top_ks}
    expected_uniform_by_k: dict[int, list[float]] = {k: [] for k in top_ks}
    n_concept_sentences = 0
    skipped_by_concept = 0

    for i, ex in enumerate(examples):
        sid = ex["slide_id"]
        vision, kept_idx = _load_vision(emb_dir / f"{sid}.h5", max_vision_tokens, dev)
        bcss = _load_bcss(bcss_dir / f"{sid}.h5", kept_idx)
        out = generate_with_attention(
            model, vision_tokens=vision, instruction=instruction, max_new_tokens=max_new_tokens,
        )

        slide_hits: dict[int, int] = {k: 0 for k in top_ks}
        slide_concept_sents = 0
        slide_skipped = 0
        for sent, attn in zip(out["sentences"], out["per_sentence_attn"], strict=False):
            channel = _detect_concept(sent)
            if channel is None:
                slide_skipped += 1
                continue
            mask = bcss.get(f"mask_{channel}")
            roi = bcss.get("roi_coverage")
            if mask is None or roi is None or roi.sum() == 0:
                slide_skipped += 1
                continue
            slide_concept_sents += 1

            ranked = np.argsort(attn)[::-1]
            target = (mask >= BCSS_THRESHOLD) & (roi > 0)
            n_target = int(target.sum())
            n_patches = attn.size
            for k in top_ks:
                kk = min(k, n_patches)
                topk = ranked[:kk]
                hit = bool(target[topk].any())
                slide_hits[k] += int(hit)
                # Uniform-random baseline P(hit) = 1 - C(N - T, k) / C(N, k)
                if n_target > 0 and kk <= n_patches:
                    # complement: all k miss
                    miss_p = 1.0
                    for j in range(kk):
                        miss_p *= max(n_patches - n_target - j, 0) / max(n_patches - j, 1)
                    expected_uniform_by_k[k].append(1.0 - miss_p)
                else:
                    expected_uniform_by_k[k].append(0.0)

        n_concept_sentences += slide_concept_sents
        skipped_by_concept += slide_skipped
        for k in top_ks:
            hits_by_k[k].append(slide_hits[k])

        per_slide.append({
            "slide_id": sid,
            "n_sentences": len(out["sentences"]),
            "n_concept_sentences": slide_concept_sents,
            "n_skipped": slide_skipped,
            "hits_by_k": slide_hits,
        })
        logger.info(
            "[%d/%d] %s: concept_sents=%d skipped=%d hits@1=%d hits@5=%d hits@10=%d",
            i + 1, len(examples), sid, slide_concept_sents, slide_skipped,
            slide_hits.get(1, 0), slide_hits.get(5, 0), slide_hits.get(10, 0),
        )

    corpus: dict[str, float | int] = {
        "n_slides": len(per_slide),
        "n_concept_sentences": n_concept_sentences,
        "n_skipped_no_concept": skipped_by_concept,
    }
    total_hits = {k: sum(hits_by_k[k]) for k in top_ks}
    for k in top_ks:
        corpus[f"PG@{k}"] = (total_hits[k] / n_concept_sentences) if n_concept_sentences else 0.0
        corpus[f"PG_uniform@{k}"] = (
            float(np.mean(expected_uniform_by_k[k])) if expected_uniform_by_k[k] else 0.0
        )

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"corpus": corpus, "per_slide": per_slide, "checkpoint": checkpoint,
         "test_set": test_set, "bcss_threshold": BCSS_THRESHOLD, "top_ks": list(top_ks)},
        open(out_path, "w"),
        indent=2,
    )
    logger.info("Wrote %s — PG@1=%.3f PG@5=%.3f PG@10=%.3f  (uniform: %.3f / %.3f / %.3f)",
                out_path, corpus["PG@1"], corpus["PG@5"], corpus["PG@10"],
                corpus["PG_uniform@1"], corpus["PG_uniform@5"], corpus["PG_uniform@10"])


if __name__ == "__main__":
    app()
