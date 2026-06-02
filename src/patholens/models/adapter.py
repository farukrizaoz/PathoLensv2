"""Linear projection adapter (LLaVA-style).

Maps TITAN slide tokens (768-dim) to Llama hidden dim (3072 for Llama-3.2-3B).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class LinearAdapter(nn.Module):
    """Single linear projection from vision space to LLM space.

    Args:
        in_dim: vision token dimension (TITAN: 768)
        out_dim: LLM hidden dimension (Llama-3.2-3B: 3072)
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 3072) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, slide_tokens: Tensor) -> Tensor:
        """
        Args:
            slide_tokens: (B, N_tokens, in_dim) or (N_tokens, in_dim)
                — any dtype; internally cast to the projection's weight dtype
                so callers can pass bf16 (inference) or fp32 (autocast disabled)
                without per-call casting.

        Returns:
            (B, N_tokens, out_dim) or (N_tokens, out_dim) — same dtype as weight.
        """
        if slide_tokens.dtype != self.proj.weight.dtype:
            slide_tokens = slide_tokens.to(self.proj.weight.dtype)
        return self.proj(slide_tokens)
