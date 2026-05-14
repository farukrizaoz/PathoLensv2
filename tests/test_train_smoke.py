"""Smoke tests for GroundedVLM + training infrastructure.

All tests run without GPU, without HuggingFace dependencies, and without data files.
The LLM is replaced by a MockLLM that returns correct tensor shapes.

Design pattern: inject the mock via `model.llm = mock` — this works because
grounded_vlm.py imports transformers/peft lazily inside .load(), so the module
imports cleanly without those packages.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from patholens.models.grounded_vlm import GroundedVLM

# ─── MockLLM ──────────────────────────────────────────────────────────────────

NUM_LAYERS = 28  # Llama-3.2-3B has 28 transformer layers
NUM_HEADS = 4  # reduced for test speed


class MockLLM(nn.Module):
    """Minimal fake Llama that returns plausible tensor shapes.

    Designed for injecting into GroundedVLM via `model.llm = MockLLM()`.
    All operations are differentiable so gradient flow tests work correctly.

    Args:
        hidden_dim: LLM hidden size (must match adapter out_dim)
        vocab_size: vocabulary size (matches Llama-3.2 default 32000)
    """

    def __init__(self, hidden_dim: int = 3072, vocab_size: int = 32000) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        # Embedding table and a tiny output head so loss is differentiable
        self._embed = nn.Embedding(vocab_size, hidden_dim)
        self._head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        """Standard HuggingFace API — GroundedVLM calls this to embed text tokens."""
        return self._embed

    def forward(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        """Forward pass returning a SimpleNamespace that mirrors HF model output.

        Args:
            inputs_embeds: (B, seq_len, hidden_dim)
            attention_mask: (B, seq_len) — not used in mock
            labels: (B, seq_len) — used only to trigger loss computation
            output_attentions: if True, generate fake attention tensors
            return_dict: ignored (always returns SimpleNamespace)

        Returns:
            SimpleNamespace with .loss, .logits, .attentions
        """
        B, seq_len, _ = inputs_embeds.shape

        # Logits depend on inputs_embeds so gradients flow through the adapter
        logits: Tensor = self._head(inputs_embeds)  # (B, seq_len, vocab_size)

        loss: Tensor | None = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_logits.view(-1, self.vocab_size).detach().argmax(dim=-1),
            )

        attentions: list[Tensor] | None = None
        if output_attentions:
            # Fake attention: (B, NUM_HEADS, seq_len, seq_len), softmax-normalised
            raw = torch.randn(B, NUM_HEADS, seq_len, seq_len)
            softmaxed = torch.softmax(raw, dim=-1)
            attentions = [softmaxed] * NUM_LAYERS

        return SimpleNamespace(loss=loss, logits=logits, attentions=attentions)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

N_V = 16  # number of vision tokens (patches)
T = 20  # text sequence length
B = 1  # batch size
VOCAB = 32000
VISION_DIM = 768
LLM_DIM = 3072


@pytest.fixture
def vlm_with_mock() -> GroundedVLM:  # noqa: F821 (forward ref resolved at runtime)
    """GroundedVLM with injected MockLLM — no HuggingFace dependencies."""
    from patholens.models.grounded_vlm import GroundedVLM

    model = GroundedVLM(
        llm_repo="meta-llama/Llama-3.2-3B-Instruct",
        vision_dim=VISION_DIM,
        llm_hidden_dim=LLM_DIM,
    )
    model.llm = MockLLM(hidden_dim=LLM_DIM, vocab_size=VOCAB)
    return model


# ─── Import tests ─────────────────────────────────────────────────────────────


def test_grounded_vlm_imports_without_hf() -> None:
    """GroundedVLM can be imported without transformers / peft installed."""
    from patholens.models.grounded_vlm import GroundedVLM

    assert GroundedVLM is not None


def test_grounded_vlm_construction() -> None:
    """GroundedVLM.__init__ succeeds without calling .load() (no HF needed)."""
    from patholens.models.grounded_vlm import GroundedVLM

    model = GroundedVLM()
    assert model.llm is None  # lazy; .load() not called
    assert model.adapter is not None


# ─── Forward pass ─────────────────────────────────────────────────────────────


def test_forward_returns_logits(vlm_with_mock: GroundedVLM) -> None:
    """Forward without labels returns logits with correct shape."""
    vision = torch.randn(B, N_V, VISION_DIM)
    ids = torch.zeros(B, T, dtype=torch.long)
    mask = torch.ones(B, T, dtype=torch.long)

    out = vlm_with_mock(vision_tokens=vision, input_ids=ids, attention_mask=mask)

    assert "logits" in out
    assert out["logits"].shape == (B, N_V + T, VOCAB)
    assert out["loss"] is None
    assert out["n_vision_tokens"] == N_V


def test_forward_computes_loss_with_labels(vlm_with_mock: GroundedVLM) -> None:
    """Forward with labels returns a scalar finite loss."""
    vision = torch.randn(B, N_V, VISION_DIM)
    ids = torch.randint(0, VOCAB, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    labels = ids.clone()
    labels[:, :5] = -100  # mask first 5 positions

    out = vlm_with_mock(
        vision_tokens=vision,
        input_ids=ids,
        attention_mask=mask,
        labels=labels,
    )

    assert out["loss"] is not None
    assert out["loss"].ndim == 0
    assert not torch.isnan(out["loss"])


def test_forward_attention_extraction(vlm_with_mock: GroundedVLM) -> None:
    """output_attentions=True yields text_to_vision_attn with correct shape."""
    vision = torch.randn(B, N_V, VISION_DIM)
    ids = torch.zeros(B, T, dtype=torch.long)
    mask = torch.ones(B, T, dtype=torch.long)

    out = vlm_with_mock(
        vision_tokens=vision,
        input_ids=ids,
        attention_mask=mask,
        output_attentions=True,
    )

    assert "text_to_vision_attn" in out
    attn = out["text_to_vision_attn"]
    # (B, T, N_v)
    assert attn.shape == (B, T, N_V), f"Expected ({B}, {T}, {N_V}), got {attn.shape}"


def test_forward_no_attn_when_flag_false(vlm_with_mock: GroundedVLM) -> None:
    """text_to_vision_attn absent when output_attentions=False (default)."""
    vision = torch.randn(B, N_V, VISION_DIM)
    ids = torch.zeros(B, T, dtype=torch.long)
    mask = torch.ones(B, T, dtype=torch.long)

    out = vlm_with_mock(
        vision_tokens=vision,
        input_ids=ids,
        attention_mask=mask,
        output_attentions=False,
    )

    assert "text_to_vision_attn" not in out


# ─── Gradient flow ────────────────────────────────────────────────────────────


def test_adapter_receives_gradients(vlm_with_mock: GroundedVLM) -> None:
    """Adapter parameters must have non-zero gradients after a backward pass."""
    vision = torch.randn(B, N_V, VISION_DIM, requires_grad=False)
    ids = torch.randint(0, VOCAB, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    labels = ids.clone()

    vlm_with_mock.adapter.zero_grad()

    out = vlm_with_mock(
        vision_tokens=vision,
        input_ids=ids,
        attention_mask=mask,
        labels=labels,
    )

    assert out["loss"] is not None
    out["loss"].backward()

    adapter_grads = [p.grad for p in vlm_with_mock.adapter.parameters() if p.grad is not None]
    assert len(adapter_grads) > 0, "No adapter parameter received a gradient"
    assert any(g.abs().sum() > 0 for g in adapter_grads), "All adapter gradients are zero"


def test_no_gradient_on_frozen_base(vlm_with_mock: GroundedVLM) -> None:
    """MockLLM head and embed are trainable here, but base model would be frozen.

    This verifies the forward pass does not error when requires_grad is mixed.
    """
    # Freeze mock embedding to simulate frozen base
    for param in vlm_with_mock.llm._embed.parameters():
        param.requires_grad_(False)

    vision = torch.randn(B, N_V, VISION_DIM)
    ids = torch.randint(0, VOCAB, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    labels = ids.clone()

    out = vlm_with_mock(
        vision_tokens=vision,
        input_ids=ids,
        attention_mask=mask,
        labels=labels,
    )
    out["loss"].backward()  # must not crash


# ─── CombinedLoss backward ────────────────────────────────────────────────────


def test_combined_loss_backward_kl() -> None:
    """CombinedLoss KL variant backward pass produces finite gradients."""
    from patholens.training.losses import CombinedLoss

    loss_fn = CombinedLoss(
        lambda_grounding=0.3,
        lambda_faithfulness=0.1,
        grounding_loss_type="kl",
    )

    logits = torch.randn(B, N_V + T, VOCAB, requires_grad=True)
    labels_full = torch.randint(0, VOCAB, (B, N_V + T))
    labels_full[:, :N_V] = -100  # vision positions masked
    gen_attn = torch.randn(3, N_V, requires_grad=True)
    gt_attn = torch.softmax(torch.randn(3, N_V), dim=-1)

    result = loss_fn(logits=logits, labels=labels_full, generated_attn=gen_attn, gt_attn=gt_attn)

    assert torch.isfinite(result["total"])
    result["total"].backward()

    assert logits.grad is not None
    assert gen_attn.grad is not None


def test_combined_loss_backward_cosine() -> None:
    """CombinedLoss cosine variant backward pass produces finite gradients."""
    from patholens.training.losses import CombinedLoss

    loss_fn = CombinedLoss(
        lambda_grounding=0.3,
        lambda_faithfulness=0.1,
        grounding_loss_type="cosine",
    )

    logits = torch.randn(B, N_V + T, VOCAB, requires_grad=True)
    labels_full = torch.randint(0, VOCAB, (B, N_V + T))
    labels_full[:, :N_V] = -100
    gen_attn = torch.randn(3, N_V, requires_grad=True)
    gt_attn = torch.softmax(torch.randn(3, N_V), dim=-1)

    result = loss_fn(logits=logits, labels=labels_full, generated_attn=gen_attn, gt_attn=gt_attn)

    assert torch.isfinite(result["total"])
    result["total"].backward()
    assert gen_attn.grad is not None


# ─── End-to-end one step ──────────────────────────────────────────────────────


def test_one_training_step(vlm_with_mock: GroundedVLM) -> None:
    """Simulate one full training step: forward → loss → backward → step.

    Verifies:
        - Loss is finite
        - Adapter parameters are updated (different after step)
        - No RuntimeError during backward
    """
    from patholens.training.losses import CombinedLoss

    optimizer = torch.optim.AdamW(
        [p for p in vlm_with_mock.adapter.parameters() if p.requires_grad],
        lr=1e-4,
    )

    before = [p.data.clone() for p in vlm_with_mock.adapter.parameters()]

    vision = torch.randn(B, N_V, VISION_DIM)
    ids = torch.randint(0, VOCAB, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    labels = ids.clone()

    out = vlm_with_mock(
        vision_tokens=vision,
        input_ids=ids,
        attention_mask=mask,
        labels=labels,
        output_attentions=True,
    )

    gen_attn = out["text_to_vision_attn"].mean(dim=1)  # (B, N_V) → treat batch as sentences
    gt_attn = torch.softmax(torch.randn_like(gen_attn), dim=-1)

    loss_fn = CombinedLoss(lambda_grounding=0.3, lambda_faithfulness=0.1)
    loss_dict = loss_fn(
        logits=out["logits"],
        labels=torch.cat([torch.full((B, N_V), -100, dtype=torch.long), labels], dim=1),
        generated_attn=gen_attn,
        gt_attn=gt_attn,
    )

    optimizer.zero_grad()
    loss_dict["total"].backward()
    optimizer.step()

    after = [p.data.clone() for p in vlm_with_mock.adapter.parameters()]
    changed = any(not torch.equal(b, a) for b, a in zip(before, after, strict=False))
    assert changed, "Adapter parameters did not change after optimizer step"
    assert torch.isfinite(loss_dict["total"])
