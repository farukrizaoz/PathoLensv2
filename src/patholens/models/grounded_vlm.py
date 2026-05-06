"""End-to-end Grounded VLM.

Architecture:
    slide_tokens (TITAN, frozen)
      → adapter (trainable)
      → Llama-3.2-3B-Instruct + LoRA (trainable)
      → generated text + attention maps

To be implemented in Hafta 2.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from patholens.models.adapter import LinearAdapter


class GroundedVLM(nn.Module):
    """End-to-end model: slide_tokens → text + grounding heatmaps."""

    def __init__(
        self,
        llm_repo: str = "meta-llama/Llama-3.2-3B-Instruct",
        vision_dim: int = 768,
        llm_hidden_dim: int = 3072,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_4bit: bool = True,
    ) -> None:
        super().__init__()
        self.adapter = LinearAdapter(in_dim=vision_dim, out_dim=llm_hidden_dim)
        # TODO: load Llama with bitsandbytes 4-bit + apply LoRA via PEFT
        # from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        # from peft import LoraConfig, get_peft_model, TaskType
        # self.llm = AutoModelForCausalLM.from_pretrained(...)
        # self.llm = get_peft_model(self.llm, lora_config)
        raise NotImplementedError("Implement in Hafta 2 — see docs/ARCHITECTURE.md")

    def forward(
        self,
        slide_tokens: Tensor,
        instruction_input_ids: Tensor,
        labels: Tensor | None = None,
    ) -> dict:
        """Forward pass with optional language modeling loss."""
        raise NotImplementedError

    @staticmethod
    def from_pretrained(checkpoint_path: str) -> "GroundedVLM":
        """Load from saved checkpoint."""
        raise NotImplementedError
