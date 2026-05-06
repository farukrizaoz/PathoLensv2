"""Main training entry point.

Usage:
    uv run python -m patholens.training.train --config configs/caption_baseline.yaml --run-name baseline_v1

Implemented in Hafta 3.
"""

from __future__ import annotations

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
    run_name: str = typer.Option(..., help="W&B run name"),
    resume: str | None = typer.Option(None, help="Path to checkpoint to resume from"),
) -> None:
    """Run training."""
    cfg = load_config(config)

    logger.info(f"Starting training run: {run_name}")
    logger.info(f"Config: {config}")

    # TODO (Hafta 3):
    # 1. Initialize wandb
    # 2. Load model (GroundedVLM)
    # 3. Load datasets (SlideInstructionDataset)
    # 4. Set up optimizer (AdamW with separate lr for adapter vs LoRA)
    # 5. Set up scheduler (cosine with warmup)
    # 6. Training loop with gradient accumulation
    # 7. Validation every N steps
    # 8. Save checkpoint to checkpoints/{run_name}/
    # 9. Log to wandb

    raise NotImplementedError("Implement in Hafta 3")


if __name__ == "__main__":
    app()
