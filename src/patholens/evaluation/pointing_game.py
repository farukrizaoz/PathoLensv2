"""CAMELYON16 pointing game evaluation.

For each generated sentence containing tumor-related terms (e.g., "metastasis", "carcinoma"):
    1. Extract per-sentence attention heatmap
    2. Get top-K attended patches
    3. Check if any of top-K patches overlap with CAMELYON16 tumor mask
    4. PG@K = fraction of sentences with successful pointing

To be implemented in Hafta 4.
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
    output_json: str = typer.Option("results/pointing_game.json"),
) -> None:
    """Run pointing game evaluation."""
    logger.info(f"Evaluating {checkpoint} on CAMELYON16")
    raise NotImplementedError("Implement in Hafta 4")


if __name__ == "__main__":
    app()
