"""PathoLens CLI entry point.

Registered as `patholens` command via pyproject.toml [project.scripts].
"""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="patholens",
    help="PathoLens-VLM: Spatially-grounded pathology report generation.",
    add_completion=False,
)
console = Console()


@app.command()
def info() -> None:
    """Show project info and verify installation."""
    console.print("[bold green]PathoLens-VLM[/bold green] v0.1.0")
    console.print("Spatially-grounded VLM for whole-slide breast cancer histopathology.")
    console.print()

    # Check key dependencies
    checks: list[tuple[str, str]] = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("peft", "PEFT (LoRA)"),
        ("h5py", "HDF5"),
        ("wandb", "Weights & Biases"),
    ]
    for module, display in checks:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "?")
            console.print(f"  ✓ {display} {version}")
        except ImportError:
            console.print(f"  ✗ {display} [red]not installed[/red]")


if __name__ == "__main__":
    app()
