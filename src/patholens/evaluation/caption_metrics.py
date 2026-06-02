"""Caption-quality evaluation for the grounded VLM.

For each slide in the test split:
    1. Load CONCHv1.5 patch embeddings from the HDF5 file
    2. Build a chat-formatted instruction prompt and tokenize
    3. Generate a report with GroundedVLM.generate()
    4. Compare prediction vs ground-truth caption with BLEU + ROUGE-L

Writes a JSON file with per-slide predictions + corpus-level scores so the smoke
run can be inspected in a notebook / report. Designed to be runnable even with
just a handful of slides (e.g. the 10-slide smoke set).

Usage:
    uv run python -m patholens.evaluation.caption_metrics \\
        --checkpoint checkpoints/grounded_smoke/latest \\
        --test-set data/processed/slideinstruction_smoke/test.json \\
        --embeddings-dir data/processed/embeddings/tcga_brca \\
        --output-json results/caption_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
import typer

from patholens.models.grounded_vlm import GroundedVLM
from patholens.utils.logging import setup_logger

app = typer.Typer()
logger = setup_logger(__name__)

DEFAULT_INSTRUCTION = "Generate a structured pathology report for this whole-slide image."


# ─── Metric helpers (lightweight, no heavy deps) ──────────────────────────────


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def _ngrams(toks: list[str], n: int) -> list[tuple]:
    return [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]


def _bleu(pred: str, ref: str, max_n: int = 4) -> float:
    """Sentence-level BLEU-{1..max_n} with brevity penalty (smoothed)."""
    p_toks, r_toks = _tokens(pred), _tokens(ref)
    if not p_toks or not r_toks:
        return 0.0
    import math
    from collections import Counter

    log_ps = []
    for n in range(1, max_n + 1):
        p_ng = Counter(_ngrams(p_toks, n))
        r_ng = Counter(_ngrams(r_toks, n))
        overlap = sum((p_ng & r_ng).values())
        total = max(sum(p_ng.values()), 1)
        # Add-1 smoothing so 0-overlap higher-order grams don't zero BLEU
        log_ps.append(math.log((overlap + 1) / (total + 1)))
    bp = 1.0 if len(p_toks) > len(r_toks) else math.exp(1 - len(r_toks) / len(p_toks))
    return float(bp * math.exp(sum(log_ps) / max_n))


def _rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F1 (LCS-based)."""
    p_toks, r_toks = _tokens(pred), _tokens(ref)
    if not p_toks or not r_toks:
        return 0.0
    # LCS DP
    m, n = len(p_toks), len(r_toks)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = (
                dp[i][j] + 1 if p_toks[i] == r_toks[j] else max(dp[i][j + 1], dp[i + 1][j])
            )
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / m, lcs / n
    return float(2 * prec * rec / (prec + rec))


# ─── Main ────────────────────────────────────────────────────────────────────


def _load_vision_tokens(
    h5_path: Path, max_vision_tokens: int, device: str
) -> torch.Tensor:
    with h5py.File(h5_path, "r") as f:
        emb = f["patch_embeddings_v15"][:]  # (N, 768)
    if emb.shape[0] > max_vision_tokens:
        idx = np.linspace(0, emb.shape[0] - 1, max_vision_tokens).astype(np.int64)
        emb = emb[idx]
    return torch.from_numpy(emb).to(device=device, dtype=torch.bfloat16).unsqueeze(0)


@app.command()
def main(
    checkpoint: str = typer.Option(..., help="Directory written by GroundedVLM.save_pretrained"),
    test_set: str = typer.Option(..., help="Path to test.json (slide_id/caption/instruction)"),
    embeddings_dir: str = typer.Option(
        "data/processed/embeddings/tcga_brca", help="Directory of per-slide HDF5 files"
    ),
    output_json: str = typer.Option("results/caption_metrics.json"),
    max_vision_tokens: int = typer.Option(1024),
    max_new_tokens: int = typer.Option(384),
    device: str = typer.Option("cuda"),
    limit: int = typer.Option(0, help="If >0, only evaluate the first N slides"),
) -> None:
    """Run caption-quality evaluation on a slide-level test split."""
    examples = json.load(open(test_set))
    if limit > 0:
        examples = examples[:limit]
    emb_dir = Path(embeddings_dir)
    examples = [e for e in examples if (emb_dir / f"{e['slide_id']}.h5").exists()]
    logger.info("Evaluating %d slides", len(examples))
    if not examples:
        raise SystemExit(f"No test slides have embeddings under {emb_dir}")

    logger.info("Loading checkpoint from %s", checkpoint)
    model = GroundedVLM.from_pretrained(checkpoint).to(device)
    model.eval()
    tok = model.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows: list[dict] = []
    for i, ex in enumerate(examples):
        sid = ex["slide_id"]
        instruction = ex.get("instruction") or DEFAULT_INSTRUCTION
        ref = ex["caption"]
        vision = _load_vision_tokens(emb_dir / f"{sid}.h5", max_vision_tokens, device)
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": instruction}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        out_ids = model.generate(
            vision_tokens=vision,
            prompt_input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        # generate() with inputs_embeds returns only new tokens
        pred = tok.decode(out_ids[0], skip_special_tokens=True).strip()
        bleu = _bleu(pred, ref)
        rouge = _rouge_l(pred, ref)
        rows.append(
            {
                "slide_id": sid,
                "instruction": instruction,
                "reference": ref,
                "prediction": pred,
                "bleu4": bleu,
                "rouge_l": rouge,
            }
        )
        logger.info("[%d/%d] %s  BLEU=%.3f  ROUGE-L=%.3f", i + 1, len(examples), sid, bleu, rouge)

    corpus = {
        "n_slides": len(rows),
        "bleu4_mean": float(np.mean([r["bleu4"] for r in rows])) if rows else 0.0,
        "rouge_l_mean": float(np.mean([r["rouge_l"] for r in rows])) if rows else 0.0,
    }
    out = {"corpus": corpus, "per_slide": rows, "checkpoint": checkpoint, "test_set": test_set}
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    logger.info("Wrote %s — BLEU4=%.3f ROUGE-L=%.3f", out_path, corpus["bleu4_mean"], corpus["rouge_l_mean"])


if __name__ == "__main__":
    app()
