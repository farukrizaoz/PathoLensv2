"""Configuration loading utilities using OmegaConf.

Loads YAML configs and provides attribute-style access.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config file and return an OmegaConf DictConfig.

    Args:
        path: path to YAML config file

    Returns:
        OmegaConf DictConfig with attribute-style access

    Raises:
        FileNotFoundError: if config file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"Expected DictConfig, got {type(cfg)}")
    return cfg


def merge_configs(base_path: str | Path, override_path: str | Path) -> DictConfig:
    """Deep-merge two YAML configs (base + override).

    Useful for grounding configs that inherit from caption_baseline.

    Args:
        base_path: path to base config
        override_path: path to override config

    Returns:
        merged DictConfig
    """
    base = load_config(base_path)
    override = load_config(override_path)
    return OmegaConf.merge(base, override)
