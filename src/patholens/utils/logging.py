"""Logging setup for PathoLens.

Provides a consistent logger with rich formatting across all modules.
"""

from __future__ import annotations

import logging
import sys

from rich.logging import RichHandler


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger with Rich formatting.

    Args:
        name: logger name (typically __name__)
        level: logging level

    Returns:
        configured Logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=True,
    )
    handler.setLevel(level)

    fmt = logging.Formatter("%(message)s", datefmt="[%X]")
    handler.setFormatter(fmt)

    logger.addHandler(handler)
    logger.propagate = False

    return logger
