"""Embedding precompute pipeline.

For each WSI:
    1. Tissue segmentation (Otsu / morphological fallback)
    2. Patch extraction (448×448 @20×, tissue ratio > 0.5)
    3. CONCHv1.5 patch encoding → (N_patches, 768) float16
    4. CONCH v1 patch encoding   → (N_patches, 512) float16
    5. TITAN slide encoding      → (N_slide_tokens, 768) float16
    6. HDF5 cache per slide
    7. QC checks; corrupt slides are skipped and logged to a JSON report

Usage:
    uv run python -m patholens.data.precompute_embeddings --config configs/precompute.yaml
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision.transforms as T
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from patholens.data.wsi_preprocessing import (
    extract_patches,
    get_tissue_mask,
)
from patholens.models.conch_encoder import ConchV1Encoder
from patholens.models.titan_encoder import TITANEncoderWrapper
from patholens.utils.config import load_config
from patholens.utils.logging import setup_logger

app = typer.Typer()
console = Console()
logger = setup_logger(__name__)

# ─── QC thresholds (from DATA.md) ────────────────────────────────────────────
QC_MIN_PATCHES = 1_000
QC_MAX_PATCHES = 15_000
QC_MAX_HDF5_MB = 200.0  # generous upper bound; typical is 5–30 MB
QC_NORM_TOL = 0.2  # |‖embedding‖ − 1| < 0.2 (L2-normalised encoders)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ─── Preprocessing transform ─────────────────────────────────────────────────


def _build_transform() -> T.Compose:
    """Standard 448×448 ImageNet normalisation used by both CONCH variants."""
    return T.Compose(
        [
            T.ToTensor(),  # uint8 HWC → float32 CHW, [0, 1]
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


# ─── Batch encoding helper ───────────────────────────────────────────────────


def _embed_patches_batched(
    patches_np: np.ndarray,
    encode_fn: object,
    transform: T.Compose,
    batch_size: int,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
) -> np.ndarray:
    """Encode (N, H, W, 3) uint8 patches to (N, D) float32 via batched forward pass.

    Args:
        patches_np: (N, H, W, 3) uint8 RGB array
        encode_fn: callable(Tensor) → Tensor — already-loaded encoder method
        transform: preprocessing transform applied per patch
        batch_size: patches per forward pass
        device: torch device string
        dtype: compute dtype for the encoder

    Returns:
        (N, D) float32 numpy array
    """
    all_embs: list[np.ndarray] = []
    n = len(patches_np)

    for start in range(0, n, batch_size):
        batch_np = patches_np[start : start + batch_size]  # (B, H, W, 3) uint8
        # PIL-compatible path: transform expects HWC uint8 numpy → CHW float
        from PIL import Image

        tensors = [transform(Image.fromarray(patch)) for patch in batch_np]
        batch_t = torch.stack(tensors).to(device, dtype=dtype)  # (B, 3, H, W)
        emb = encode_fn(batch_t)
        all_embs.append(emb.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0) if all_embs else np.empty((0, 0), dtype=np.float32)


# ─── QC ──────────────────────────────────────────────────────────────────────


def _qc_check(
    slide_id: str,
    patches: np.ndarray,
    emb_v15: np.ndarray,
    emb_v1: np.ndarray,
    slide_tokens: np.ndarray,
    coordinates: np.ndarray,
) -> list[str]:
    """Return a list of QC failure strings (empty = pass)."""
    issues: list[str] = []
    n = len(patches)

    if n < QC_MIN_PATCHES:
        issues.append(f"patch_count={n} below QC_MIN={QC_MIN_PATCHES}")
    if n > QC_MAX_PATCHES:
        issues.append(f"patch_count={n} above QC_MAX={QC_MAX_PATCHES}")

    for name, arr in [("emb_v15", emb_v15), ("emb_v1", emb_v1), ("slide_tokens", slide_tokens)]:
        if np.any(np.isnan(arr)):
            issues.append(f"{name} contains NaN")
        if np.any(np.isinf(arr)):
            issues.append(f"{name} contains Inf")

    # Embedding norms should be close to 1 (both CONCH encoders L2-normalise)
    for name, arr in [("emb_v15", emb_v15), ("emb_v1", emb_v1)]:
        if len(arr) > 0:
            norms = np.linalg.norm(arr, axis=-1)
            mean_norm = float(norms.mean())
            if abs(mean_norm - 1.0) > QC_NORM_TOL:
                issues.append(f"{name} mean norm={mean_norm:.3f} (expected ≈1.0)")

    if slide_tokens.ndim != 2 or slide_tokens.shape[1] != 768:
        issues.append(f"slide_tokens shape={slide_tokens.shape}, expected (M, 768)")

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        issues.append(f"coordinates shape={coordinates.shape}, expected (N, 2)")

    return issues


# ─── Per-slide processor ─────────────────────────────────────────────────────


def _process_slide(
    wsi_path: Path,
    titan: TITANEncoderWrapper,
    conch_v1: ConchV1Encoder,
    output_dir: Path,
    cfg: object,
    device: str,
    transform: T.Compose,
) -> dict:
    """Process one WSI end-to-end; return a result dict with qc_issues list.

    Returns:
        dict with keys: slide_id, n_patches, n_slide_tokens, qc_issues, duration_s
    """
    t0 = time.perf_counter()
    slide_id = wsi_path.stem
    result: dict = {"slide_id": slide_id, "n_patches": 0, "n_slide_tokens": 0, "qc_issues": []}

    # ── 1. Tissue mask ────────────────────────────────────────────────────────
    import openslide

    wsi = openslide.OpenSlide(str(wsi_path))
    tissue_mask, mask_downsample = get_tissue_mask(
        wsi,
        sat_threshold=cfg.sat_threshold,
        min_tissue_area_pct=cfg.min_tissue_area_pct,
        thumbnail_mag=cfg.otsu_thumbnail_mag,
    )
    wsi.close()

    # ── 2. Patch extraction ───────────────────────────────────────────────────
    patches, coordinates = extract_patches(
        wsi_path=wsi_path,
        tissue_mask=tissue_mask,
        mask_downsample=mask_downsample,
        patch_size=cfg.patch_size,
        magnification=cfg.magnification,
        tissue_ratio_threshold=cfg.tissue_ratio_threshold,
    )

    if len(patches) == 0:
        result["qc_issues"].append("no tissue patches extracted")
        result["duration_s"] = time.perf_counter() - t0
        return result

    result["n_patches"] = len(patches)

    # ── 3. CONCHv1.5 patch embeddings ─────────────────────────────────────────
    emb_v15 = _embed_patches_batched(
        patches_np=patches,
        encode_fn=titan.encode_patches,
        transform=transform,
        batch_size=cfg.batch_size,
        device=device,
        dtype=torch.bfloat16,
    )  # (N, 768)

    # ── 4. CONCH v1 patch embeddings ──────────────────────────────────────────
    emb_v1 = _embed_patches_batched(
        patches_np=patches,
        encode_fn=conch_v1.encode_patches,
        transform=transform,
        batch_size=cfg.batch_size,
        device=device,
        dtype=torch.float32,
    )  # (N, 512)

    # ── 5. TITAN slide encoding ───────────────────────────────────────────────
    emb_v15_t = torch.from_numpy(emb_v15).to(device)
    coords_t = torch.from_numpy(coordinates).to(device)
    slide_tokens = titan.encode_slide(emb_v15_t, coords_t).cpu().numpy()  # (M, 768)
    result["n_slide_tokens"] = len(slide_tokens)

    # ── 6. QC ─────────────────────────────────────────────────────────────────
    qc_issues = _qc_check(slide_id, patches, emb_v15, emb_v1, slide_tokens, coordinates)
    result["qc_issues"] = qc_issues
    if qc_issues:
        logger.warning("QC issues for %s: %s", slide_id, "; ".join(qc_issues))

    # ── 7. Save HDF5 ─────────────────────────────────────────────────────────
    out_path = output_dir / f"{slide_id}.h5"
    dtype_np = np.float16 if cfg.embeddings_dtype == "float16" else np.float32
    kwargs = {
        "compression": cfg.hdf5_compression,
        "compression_opts": cfg.hdf5_compression_opts,
    }
    with h5py.File(out_path, "w") as f:
        f.create_dataset("patch_embeddings_v15", data=emb_v15.astype(dtype_np), **kwargs)
        f.create_dataset("patch_embeddings_v1", data=emb_v1.astype(dtype_np), **kwargs)
        f.create_dataset("slide_tokens", data=slide_tokens.astype(dtype_np), **kwargs)
        f.create_dataset("coordinates", data=coordinates, **kwargs)
        f.create_dataset("tissue_mask_lowres", data=tissue_mask, compression="gzip")

        f.attrs["slide_id"] = slide_id
        f.attrs["magnification"] = cfg.magnification
        f.attrs["patch_size"] = cfg.patch_size
        f.attrs["total_tissue_patches"] = len(patches)
        f.attrs["n_slide_tokens"] = len(slide_tokens)
        f.attrs["conch_v1_repo"] = "MahmoodLab/conch"
        f.attrs["conch_v15_repo"] = "MahmoodLab/conchv1_5"
        f.attrs["titan_repo"] = cfg.titan_repo

    result["hdf5_mb"] = out_path.stat().st_size / 1e6
    result["duration_s"] = time.perf_counter() - t0
    return result


# ─── CLI entrypoint ───────────────────────────────────────────────────────────


@app.command()
def main(
    config: str = typer.Option(..., help="Path to YAML config"),
    wsi_dir: str | None = typer.Option(None, help="Override WSI directory"),
    output_dir: str | None = typer.Option(None, help="Override output directory"),
    device: str = typer.Option("cuda", help="Torch device (cuda / cpu)"),
    streaming: bool = typer.Option(False, help="Delete WSI after embedding"),
    resume: bool = typer.Option(True, help="Skip slides whose HDF5 already exists"),
) -> None:
    """Run the CONCHv1.5 + CONCH v1 + TITAN embedding precompute pipeline."""
    cfg = load_config(config)

    if wsi_dir:
        cfg.wsi_dir = wsi_dir
    if output_dir:
        cfg.output_dir = output_dir
    if streaming:
        cfg.streaming = True

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wsi_paths = sorted(
        list(Path(cfg.wsi_dir).glob("*.svs")) + list(Path(cfg.wsi_dir).glob("*.tif"))
    )
    if not wsi_paths:
        console.print(f"[red]No .svs/.tif files found in {cfg.wsi_dir}[/red]")
        raise typer.Exit(1)

    # Resume: skip already-processed slides
    if resume:
        before = len(wsi_paths)
        wsi_paths = [p for p in wsi_paths if not (out_dir / f"{p.stem}.h5").exists()]
        skipped = before - len(wsi_paths)
        if skipped:
            console.print(f"[dim]Resuming — skipping {skipped} already-processed slides[/dim]")

    console.print(f"Processing {len(wsi_paths)} WSIs → {out_dir}")

    # ── Load models ───────────────────────────────────────────────────────────
    console.print("Loading TITAN (CONCHv1.5 bundled)…")
    titan = TITANEncoderWrapper(hf_repo=cfg.titan_repo, device=device).load()

    console.print("Loading CONCH v1…")
    conch_v1 = ConchV1Encoder(device=device).load()

    transform = _build_transform()

    # ── Process slides ────────────────────────────────────────────────────────
    all_results: list[dict] = []
    failed: list[str] = []

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding slides…", total=len(wsi_paths))

        for wsi_path in wsi_paths:
            progress.update(task, description=f"[cyan]{wsi_path.name}[/cyan]")
            try:
                result = _process_slide(
                    wsi_path=wsi_path,
                    titan=titan,
                    conch_v1=conch_v1,
                    output_dir=out_dir,
                    cfg=cfg,
                    device=device,
                    transform=transform,
                )
                all_results.append(result)

                n_patches = result["n_patches"]
                n_tokens = result["n_slide_tokens"]
                dur = result["duration_s"]
                issues = result["qc_issues"]
                status = "[yellow]QC WARN[/yellow]" if issues else "[green]OK[/green]"
                console.print(
                    f"  {wsi_path.name}: {n_patches} patches, "
                    f"{n_tokens} tokens, {dur:.1f}s — {status}"
                )

                if cfg.streaming and cfg.delete_wsi_after:
                    wsi_path.unlink()
                    logger.info("Deleted WSI after embedding: %s", wsi_path)

            except Exception as exc:
                logger.exception("Failed to process %s: %s", wsi_path, exc)
                failed.append(str(wsi_path))
                console.print(f"  [red]FAILED[/red] {wsi_path.name}: {exc}")

            progress.advance(task)

    # ── Summary report ────────────────────────────────────────────────────────
    n_ok = sum(1 for r in all_results if not r["qc_issues"])
    n_warn = sum(1 for r in all_results if r["qc_issues"])
    total_patches = sum(r["n_patches"] for r in all_results)
    total_dur = sum(r["duration_s"] for r in all_results)

    console.print()
    console.print(
        f"[bold]Done.[/bold] {n_ok} OK · {n_warn} QC warnings · "
        f"{len(failed)} failed · {total_patches:,} total patches · "
        f"{total_dur / 60:.1f} min"
    )

    report_path = out_dir / "precompute_report.json"
    with open(report_path, "w") as fp:
        json.dump({"results": all_results, "failed": failed}, fp, indent=2)
    console.print(f"Report written to {report_path}")

    if failed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
