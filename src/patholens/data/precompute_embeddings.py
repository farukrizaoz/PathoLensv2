"""Embedding precompute pipeline.

For each WSI:
    1. Load WSI
    2. Tissue segmentation
    3. Patch extraction (448x448 @20x)
    4. CONCHv1.5 patch encoding
    5. TITAN slide encoding
    6. Save HDF5 with patch_embeddings, slide_tokens, coordinates

Usage:
    uv run python -m patholens.data.precompute_embeddings --config configs/precompute.yaml
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from patholens.utils.config import load_config
from patholens.utils.logging import setup_logger

app = typer.Typer()
console = Console()
logger = setup_logger(__name__)


@app.command()
def main(
    config: str = typer.Option(..., help="Path to YAML config"),
    wsi_dir: str | None = typer.Option(None, help="Override WSI directory"),
    output_dir: str | None = typer.Option(None, help="Override output directory"),
    streaming: bool = typer.Option(False, help="Stream WSI download + delete after embed"),
) -> None:
    """Run embedding precompute pipeline."""
    cfg = load_config(config)

    if wsi_dir:
        cfg.wsi_dir = wsi_dir
    if output_dir:
        cfg.output_dir = output_dir

    logger.info(f"Config: {config}")
    logger.info(f"WSI dir: {cfg.wsi_dir}")
    logger.info(f"Output dir: {cfg.output_dir}")
    logger.info(f"Patch size: {cfg.patch_size}, magnification: {cfg.magnification}")

    wsi_paths = list(Path(cfg.wsi_dir).glob("*.svs")) + list(Path(cfg.wsi_dir).glob("*.tif"))
    logger.info(f"Found {len(wsi_paths)} WSI files")

    # TODO (Hafta 1):
    # for wsi_path in wsi_paths:
    #     1. tissue mask
    #     2. extract patches
    #     3. CONCHv1.5 forward → patch_embeddings
    #     4. TITAN slide encoder → slide_tokens
    #     5. save HDF5
    #     6. if streaming: rm wsi_path

    raise NotImplementedError("Implement in Hafta 1 — see docs/DATA.md")


if __name__ == "__main__":
    app()
