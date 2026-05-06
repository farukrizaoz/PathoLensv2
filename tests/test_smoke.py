"""Smoke tests — run without GPU, without data.

Usage:
    make test
    uv run pytest tests/ -v -m "not slow and not gpu"
"""

from __future__ import annotations

import pytest


# ── Imports ──


def test_import_adapter():
    """LinearAdapter module imports cleanly."""
    from patholens.models.adapter import LinearAdapter

    assert LinearAdapter is not None


def test_import_losses():
    """Loss modules import cleanly."""
    from patholens.training.losses import CaptionLoss, CombinedLoss, GroundingLoss

    assert CaptionLoss is not None
    assert GroundingLoss is not None
    assert CombinedLoss is not None


def test_import_config():
    """Config utility imports cleanly."""
    from patholens.utils.config import load_config

    assert callable(load_config)


def test_import_logging():
    """Logging utility imports cleanly."""
    from patholens.utils.logging import setup_logger

    logger = setup_logger("test")
    assert logger is not None


# ── Adapter ──


def test_adapter_forward():
    """LinearAdapter forward pass with random input."""
    import torch

    from patholens.models.adapter import LinearAdapter

    adapter = LinearAdapter(in_dim=768, out_dim=3072)
    x = torch.randn(4, 256, 768)  # (B, N_tokens, 768)
    out = adapter(x)
    assert out.shape == (4, 256, 3072)


def test_adapter_single_sample():
    """LinearAdapter works without batch dimension."""
    import torch

    from patholens.models.adapter import LinearAdapter

    adapter = LinearAdapter(in_dim=768, out_dim=3072)
    x = torch.randn(256, 768)  # (N_tokens, 768)
    out = adapter(x)
    assert out.shape == (256, 3072)


# ── Losses ──


def test_caption_loss():
    """CaptionLoss computes scalar loss."""
    import torch

    from patholens.training.losses import CaptionLoss

    loss_fn = CaptionLoss()
    logits = torch.randn(2, 10, 32000)  # (B, seq_len, vocab_size)
    labels = torch.randint(0, 32000, (2, 10))
    labels[:, :3] = -100  # mask vision positions

    loss = loss_fn(logits, labels)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0


def test_grounding_loss():
    """GroundingLoss KL divergence with random attention."""
    import torch

    from patholens.training.losses import GroundingLoss

    loss_fn = GroundingLoss(temperature=0.1)
    gen_attn = torch.randn(3, 256)  # 3 sentences, 256 vision tokens
    gt_attn = torch.randn(3, 256)

    loss = loss_fn(gen_attn, gt_attn)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_faithfulness_reg():
    """FaithfulnessRegularizer returns mean entropy."""
    import torch

    from patholens.training.losses import FaithfulnessRegularizer

    reg = FaithfulnessRegularizer()
    attn = torch.softmax(torch.randn(3, 256), dim=-1)

    entropy = reg(attn)
    assert entropy.ndim == 0
    assert entropy.item() >= 0


def test_combined_loss_caption_only():
    """CombinedLoss with grounding disabled."""
    import torch

    from patholens.training.losses import CombinedLoss

    loss_fn = CombinedLoss(lambda_grounding=0.0, lambda_faithfulness=0.0)
    logits = torch.randn(1, 10, 32000)
    labels = torch.randint(0, 32000, (1, 10))

    result = loss_fn(logits, labels)
    assert "total" in result
    assert "caption" in result
    assert "grounding" not in result


def test_combined_loss_with_grounding():
    """CombinedLoss with grounding enabled."""
    import torch

    from patholens.training.losses import CombinedLoss

    loss_fn = CombinedLoss(lambda_grounding=0.3, lambda_faithfulness=0.1)
    logits = torch.randn(1, 10, 32000)
    labels = torch.randint(0, 32000, (1, 10))
    gen_attn = torch.randn(3, 256)
    gt_attn = torch.randn(3, 256)

    result = loss_fn(logits, labels, generated_attn=gen_attn, gt_attn=gt_attn)
    assert "total" in result
    assert "caption" in result
    assert "grounding" in result
    assert "faithfulness" in result
    assert result["total"].item() > 0


# ── Config ──


def test_load_config(tmp_path):
    """load_config reads YAML correctly."""
    from patholens.utils.config import load_config

    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("batch_size: 4\nlr: 0.001\n")

    cfg = load_config(cfg_file)
    assert cfg.batch_size == 4
    assert cfg.lr == 0.001


def test_load_config_missing():
    """load_config raises on missing file."""
    from patholens.utils.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")
