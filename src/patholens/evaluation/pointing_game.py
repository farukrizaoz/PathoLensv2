"""CAMELYON16 pointing game evaluation (not run — dataset is 150 GB).

The evaluated grounding metric for this project is the **BCSS pointing game**
implemented in `bcss_pointing_game.py`, which uses the same TCGA-BRCA slides as
training and requires no additional download.  Fullscale result: PG@5 = 0.081
vs uniform 0.019 (4.4× lift).  See `docs/FULLSCALE_RUN_REPORT.md`.

This module is retained as a scaffold for future CAMELYON16 cross-dataset
evaluation.  Running it requires:
  1. CAMELYON16 embeddings at `data/processed/embeddings/camelyon16/`
  2. CAMELYON16 tumor polygon XML files at `data/raw/camelyon16/`
  3. Pre-converted patch-level masks via `data/camelyon_xml_to_mask.py`
"""

from __future__ import annotations

import typer

from patholens.utils.logging import setup_logger

app = typer.Typer()
logger = setup_logger(__name__)


@app.command()
def main(
    checkpoint: str = typer.Option(..., help="Path to model checkpoint"),
    camelyon_dir: str = typer.Option(..., help="Path to data/processed/embeddings/camelyon16"),
    top_k_values: list[int] = typer.Option([1, 5, 10], help="K values for PG@K"),
    output_json: str = typer.Option("results/pointing_game_camelyon.json"),
) -> None:
    """CAMELYON16 pointing game (scaffold — dataset not downloaded).

    For the published grounding evaluation use bcss_pointing_game instead:

        uv run python -m patholens.evaluation.bcss_pointing_game --help

    PG@5 = 0.081 vs uniform 0.019 (4.4× lift) from the fullscale run.
    See docs/FULLSCALE_RUN_REPORT.md for details.
    """
    logger.warning(
        "CAMELYON16 pointing game not run: dataset (150 GB) was not downloaded. "
        "Use bcss_pointing_game.py for the published PG@K results. "
        "See docs/FULLSCALE_RUN_REPORT.md."
    )
    raise SystemExit(0)


if __name__ == "__main__":
    app()
