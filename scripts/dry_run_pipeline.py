"""End-to-end pipeline sanity check on a single slide.

Before committing to a multi-hour training run, this script:

  1. Picks one slide from the embeddings dir (or one you specify).
  2. Verifies the H5 has the datasets the trainer expects (`patch_embeddings_v15`,
     `coordinates`, `tissue_mask_lowres`) and that arrays are sane (no NaN, right
     shapes, normalised).
  3. Verifies BCSS patch-mask file exists and has non-trivial ROI coverage.
  4. Walks a single forward + loss step exactly the way the trainer would,
     using the smoke config — so any silent dtype / shape / device drift fails
     here in <2 minutes instead of after 1 hour of real training.
  5. Loads the latest checkpoint (if any), runs `generate_with_attention`, and
     asserts: text non-empty, n_sentences == len(per_sentence_attn), each attn
     distribution sums to ≈ 1 and is finite.

Returns exit code 0 only if every check passes; otherwise prints the failing
check and exits 1 (so the overnight chain can gate on it).

Usage:
    uv run python scripts/dry_run_pipeline.py \\
        --config configs/grounding_smoke.yaml \\
        --checkpoint checkpoints/grounded_smoke_20260602_105250/final
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
import torch
import typer
from omegaconf import OmegaConf

app = typer.Typer()


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")
    raise typer.Exit(1)


@app.command()
def main(
    config: str = typer.Option("configs/grounding_smoke.yaml"),
    checkpoint: str | None = typer.Option(None, help="If set, also test inference"),
    slide_id: str | None = typer.Option(None, help="Pick a specific slide"),
    device: str = typer.Option("cuda"),
) -> None:
    cfg = OmegaConf.load(config)

    # ── 1. Pick a slide
    emb_dir = Path(cfg.embeddings_dir)
    h5s = sorted(emb_dir.glob("*.h5"))
    if not h5s:
        _fail(f"no H5 files under {emb_dir}")
    chosen = next((p for p in h5s if p.stem == slide_id), h5s[0])
    sid = chosen.stem
    print(f"\n[1] Slide: {sid}")
    _ok(f"H5 exists: {chosen}")

    # ── 2. H5 contents
    print("\n[2] H5 schema + arrays")
    with h5py.File(chosen, "r") as f:
        assert "patch_embeddings_v15" in f, "missing patch_embeddings_v15"
        assert "coordinates" in f, "missing coordinates"
        assert "tissue_mask_lowres" in f, "missing tissue_mask_lowres"
        emb = f["patch_embeddings_v15"][:]
        coords = f["coordinates"][:]
        mask = f["tissue_mask_lowres"][:]
    if emb.ndim != 2 or emb.shape[1] != 768:
        _fail(f"emb shape {emb.shape}, expected (N, 768)")
    if coords.shape[0] != emb.shape[0] or coords.shape[1] != 2:
        _fail(f"coords shape {coords.shape}, expected ({emb.shape[0]}, 2)")
    if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
        _fail("NaN/Inf in patch embeddings")
    mean_norm = float(np.linalg.norm(emb.astype(np.float32), axis=-1).mean())
    if abs(mean_norm - 1.0) > 0.1:
        _fail(f"emb mean norm {mean_norm:.3f}, expected ≈1.0 (L2-normalised)")
    _ok(f"emb (N={emb.shape[0]}, 768) mean_norm={mean_norm:.3f}")
    _ok(f"coords {coords.shape} range x=[{coords[:,0].min()}, {coords[:,0].max()}], y=[{coords[:,1].min()}, {coords[:,1].max()}]")
    _ok(f"tissue_mask_lowres {mask.shape} dtype={mask.dtype}")

    # ── 3. BCSS coverage (optional but expected for grounded training)
    print("\n[3] BCSS patch mask")
    bcss_dir = Path(cfg.get("bcss_masks_dir", "") or "")
    bcss_path = bcss_dir / f"{sid}.h5" if bcss_dir else None
    if bcss_path and bcss_path.exists():
        with h5py.File(bcss_path, "r") as f:
            roi_cov = f["roi_coverage"][:] if "roi_coverage" in f else None
        n_roi = int(roi_cov.sum()) if roi_cov is not None else 0
        if n_roi == 0:
            _fail(f"BCSS file exists but roi_coverage is all-zero for {sid}")
        _ok(f"BCSS ROI-covered patches: {n_roi}/{len(coords)}")
    else:
        print(f"  ⚠ no BCSS mask for {sid} (training will use pseudo-GT fallback)")

    # ── 4. Single trainer step (caption + grounding + faithfulness)
    print("\n[4] Single trainer forward + loss")
    try:
        from patholens.data.dataset import GroundingSlideDataset, grounding_collate_fn
        from patholens.models.grounded_vlm import GroundedVLM
        from patholens.training.losses import CombinedLoss
        from patholens.training.train import _load_bcss_masks  # type: ignore
        from patholens.training.grounding_targets import PseudoGTCache, build_target
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.llm_repo)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        ds = GroundingSlideDataset(
            instruction_json=cfg.instruction_train,
            embeddings_dir=cfg.embeddings_dir,
            tokenizer=tokenizer,
            max_vision_tokens=cfg.max_vision_tokens,
            bcss_masks_dir=cfg.get("bcss_masks_dir"),
        )
        if len(ds) == 0:
            _fail("dataset is empty (no slides have both caption + embedding)")
        _ok(f"dataset size: {len(ds)}")

        item = next(it for it in ds if it["slide_id"] == sid) if any(
            d["slide_id"] == sid for d in [ds[i] for i in range(min(len(ds), 5))]
        ) else ds[0]
        batch = grounding_collate_fn([item])
        _ok("collate_fn produced a batch")

        model = GroundedVLM(
            llm_repo=cfg.llm_repo, vision_dim=cfg.vision_dim,
            llm_hidden_dim=cfg.llm_hidden_dim, lora_r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
            lora_target_modules=list(cfg.lora.target_modules),
            load_in_4bit=cfg.get("load_in_4bit", True),
            attn_layer=cfg.get("attn_layer", 14),
        ).load(device=device)
        _ok("model loaded")

        vision = batch["patch_embeddings_v15"][0].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        out = model(
            vision_tokens=vision,
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
            output_attentions=True,
        )
        if out.get("text_to_vision_attn") is None:
            _fail("model.forward did not return text_to_vision_attn")
        _ok(f"forward OK: logits {tuple(out['logits'].shape)}, text→vision {tuple(out['text_to_vision_attn'].shape)}")

        # Combined loss with the same lambdas the trainer uses
        combined = CombinedLoss(
            lambda_grounding=float(cfg.loss.lambda_grounding),
            lambda_faithfulness=float(cfg.loss.lambda_faithfulness),
            grounding_loss_type=cfg.loss.loss_type,
        )
        # Build one fake sentence target with build_target to exercise the path
        sentences = item["sentences"]
        spans = item["sentence_spans"]
        bcss_h5 = item.get("bcss_h5_path")
        bcss_masks = _load_bcss_masks(bcss_h5, len(item["patch_embeddings_v1"]))
        cache = PseudoGTCache(cache_dir="/tmp/pgt_dryrun")
        s0 = sentences[0] if sentences else "tumor."
        gt, used = build_target(
            slide_id=sid, sentence=s0,
            patch_embeddings_v1=item["patch_embeddings_v1"],
            coordinates=item["coordinates"],
            conch_v1_encoder=None,
            bcss_masks=bcss_masks,
            pseudo_gt_cache=cache,
            grounding_source=cfg.grounding.source,
        )
        gen_attn = out["text_to_vision_attn"][0, spans[0][0]:spans[0][1], :].mean(dim=0, keepdim=True)
        gt_t = torch.from_numpy(gt).unsqueeze(0).to(device)
        n_v = out["n_vision_tokens"]
        full_labels = torch.cat([
            torch.full((1, n_v), -100, dtype=batch["labels"].dtype, device=device),
            batch["labels"].to(device),
        ], dim=1)
        loss_dict = combined(
            logits=out["logits"], labels=full_labels,
            generated_attn=gen_attn, gt_attn=gt_t,
        )
        for k, v in loss_dict.items():
            val = float(v.detach().cpu())
            if not np.isfinite(val):
                _fail(f"non-finite loss {k}={val}")
            print(f"      {k:15s} = {val:.4f}")
        # Faithfulness sanity: the *mechanism* must respond to peakiness.
        # Untrained attention is genuinely near-uniform, so observed value
        # being close to log(N_v) is expected. We verify the fix by injecting
        # a deliberately peaky distribution and confirming entropy drops far
        # below log(N_v).
        if "faithfulness" in loss_dict:
            from patholens.training.losses import FaithfulnessRegularizer
            reg = FaithfulnessRegularizer()
            f_val = float(loss_dict["faithfulness"].detach().cpu())
            uniform = float(np.log(n_v))
            peaky = torch.zeros(1, n_v, device=device)
            peaky[0, : max(1, n_v // 100)] = 1.0  # mass on top 1% of patches
            peaky_entropy = float(reg(peaky).detach().cpu())
            if peaky_entropy >= uniform - 1.0:
                _fail(
                    f"faithfulness reg does not respond to peakiness: "
                    f"peaky_entropy={peaky_entropy:.3f} vs uniform={uniform:.3f}"
                )
            _ok(
                f"faithfulness observed={f_val:.3f}, uniform={uniform:.3f}, "
                f"peaky-input gives entropy={peaky_entropy:.3f}  (mechanism is responsive)"
            )
        # Backward
        loss_dict["total"].backward()
        _ok("backward OK (no autograd error)")
    except Exception:
        traceback.print_exc()
        _fail("trainer dry-run raised")

    # ── 5. Inference + attention extraction (if checkpoint provided)
    if checkpoint:
        print(f"\n[5] Inference from checkpoint {checkpoint}")
        try:
            from patholens.app.inference import DEFAULT_INSTRUCTION, generate_with_attention
            del model  # free GPU
            torch.cuda.empty_cache()
            ckpt = GroundedVLM.from_pretrained(checkpoint).to(device)
            ckpt.eval()
            res = generate_with_attention(
                ckpt,
                vision_tokens=torch.from_numpy(emb[: cfg.max_vision_tokens]).to(
                    device=device, dtype=torch.bfloat16,
                ).unsqueeze(0),
                instruction=DEFAULT_INSTRUCTION, max_new_tokens=96,
            )
            assert res["text"].strip(), "empty generation"
            assert len(res["sentences"]) == len(res["per_sentence_attn"]), "sentence/attn mismatch"
            for a in res["per_sentence_attn"]:
                assert np.isfinite(a).all(), "NaN/Inf in attention"
                assert abs(a.sum() - 1.0) < 1e-3, f"attn sum {a.sum()}"
            _ok(f"generated {len(res['sentences'])} sentences; attention well-formed")
        except Exception:
            traceback.print_exc()
            _fail("inference dry-run raised")

    print("\nALL CHECKS PASSED ✓")


if __name__ == "__main__":
    app()
