"""Download and prepare SlideInstruction dataset.

Source: General-Medical-AI/SlideChat (HuggingFace, open).
Filters to TCGA-BRCA subset matching our 150 selected slides.
Outputs: data/processed/slideinstruction/{train,val,test}.json

Usage:
    make prepare-instruction
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main(
    output_dir: str = typer.Option(
        "data/processed/slideinstruction",
        help="Output directory",
    ),
    case_selection_csv: str = typer.Option(
        "data/metadata/case_selection.csv",
        help="CSV listing 150 selected TCGA-BRCA case IDs",
    ),
    train_frac: float = typer.Option(0.70, help="Train split fraction"),
    val_frac: float = typer.Option(0.15, help="Val split fraction"),
    seed: int = typer.Option(42, help="Random seed for split"),
) -> None:
    """Download SlideInstruction, filter to our TCGA-BRCA cases, split."""
    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load SlideInstruction from HF
    console.print("→ Loading SlideInstruction from HuggingFace...")
    # NOTE: actual repo may differ slightly, verify
    ds = load_dataset(
        "General-Medical-AI/SlideChat",
        token=os.environ.get("HF_TOKEN"),
    )
    console.print(f"  Loaded {len(ds['train'])} examples")

    # 2. Filter to our case selection
    if Path(case_selection_csv).exists():
        import pandas as pd

        cases = pd.read_csv(case_selection_csv)
        selected_case_ids = set(cases["case_id"])
        console.print(f"→ Filtering to {len(selected_case_ids)} selected cases")
        filtered = [
            ex for ex in ds["train"]
            if "-".join(ex.get("case_id", "").split("-")[:3]) in selected_case_ids
        ]
    else:
        console.print(
            f"⚠ {case_selection_csv} not found — using ALL SlideInstruction examples"
        )
        filtered = list(ds["train"])

    console.print(f"  After filter: {len(filtered)} examples")

    # 3. Split train/val/test
    random.seed(seed)
    random.shuffle(filtered)
    n = len(filtered)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    splits = {
        "train": filtered[:n_train],
        "val": filtered[n_train : n_train + n_val],
        "test": filtered[n_train + n_val :],
    }

    # 4. Save
    for split_name, examples in splits.items():
        out_file = output_path / f"{split_name}.json"
        with open(out_file, "w") as f:
            json.dump(examples, f, indent=2)
        console.print(f"  ✓ {split_name}: {len(examples)} → {out_file}")


if __name__ == "__main__":
    app()
