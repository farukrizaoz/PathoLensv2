"""End-to-end Grounded VLM.

Architecture:
    vision_tokens (CONCHv1.5 patch embeddings, frozen)
      → adapter (trainable, 768 → 3072)
      → [prepended to text embeddings]
      → Llama-3.2-3B-Instruct + LoRA r=16 (trainable)
      → (logits, text→vision attention from layer `attn_layer`)

Design note: CONCHv1.5 patch embeddings are used directly as vision tokens
(not TITAN slide tokens) so that the grounding target space (patch-level GT
from grounding_targets.py) aligns 1-to-1 with the LLM's vision token space.
TITAN slide tokens are computed separately and stored in HDF5 for potential
future use (e.g. prepending a global slide context token).

Lazy loading: transformers and peft are imported inside .load() so this
module can be imported on machines without those packages installed
(e.g. unit tests that inject a mock LLM via model.llm = mock).
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
from torch import Tensor

from patholens.models.adapter import LinearAdapter

logger = logging.getLogger(__name__)

DEFAULT_ATTN_LAYER: int = 14
LLAMA_3B_HIDDEN_DIM: int = 3072


class GroundedVLM(nn.Module):
    """Slide patch tokens → structured report + per-sentence grounding attention.

    Args:
        llm_repo: HuggingFace model id for the LLM backbone
        vision_dim: vision token dimension (CONCHv1.5: 768)
        llm_hidden_dim: LLM hidden size (Llama-3.2-3B: 3072)
        lora_r: LoRA rank
        lora_alpha: LoRA scaling factor (typically 2*r)
        lora_dropout: LoRA dropout probability
        lora_target_modules: attention projection names to apply LoRA to
        load_in_4bit: use bitsandbytes 4-bit quantization (RunPod only; set False for CPU)
        attn_layer: Llama layer index to extract grounding attention from (0-indexed)
    """

    def __init__(
        self,
        llm_repo: str = "meta-llama/Llama-3.2-3B-Instruct",
        vision_dim: int = 768,
        llm_hidden_dim: int = LLAMA_3B_HIDDEN_DIM,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        load_in_4bit: bool = True,
        attn_layer: int = DEFAULT_ATTN_LAYER,
    ) -> None:
        super().__init__()
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        self.llm_repo = llm_repo
        self.attn_layer = attn_layer

        # Always instantiate adapter (no external dependencies)
        self.adapter = LinearAdapter(in_dim=vision_dim, out_dim=llm_hidden_dim)

        # Populated by .load() or injected externally (e.g. in tests)
        self.llm: nn.Module | None = None
        self.tokenizer: object | None = None

        # Stored so from_pretrained can reconstruct the object
        self._cfg: dict = {
            "llm_repo": llm_repo,
            "vision_dim": vision_dim,
            "llm_hidden_dim": llm_hidden_dim,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": lora_target_modules,
            "load_in_4bit": load_in_4bit,
            "attn_layer": attn_layer,
        }

    # ── Factory ──────────────────────────────────────────────────────────────

    def load(self, device: str = "cuda") -> GroundedVLM:
        """Load Llama from HuggingFace and apply LoRA.

        Args:
            device: torch device string ('cuda' or 'cpu')

        Returns:
            self (for chaining)
        """
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info("Loading tokenizer: %s", self.llm_repo)
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_repo)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        cfg = self._cfg
        bnb_config = None
        if cfg["load_in_4bit"] and device != "cpu":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        logger.info("Loading %s (4-bit=%s, device=%s)", self.llm_repo, cfg["load_in_4bit"], device)
        base_llm = AutoModelForCausalLM.from_pretrained(
            self.llm_repo,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto" if device != "cpu" else None,
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["lora_target_modules"],
            bias="none",
        )
        self.llm = get_peft_model(base_llm, lora_cfg)
        self.llm.print_trainable_parameters()
        return self

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        vision_tokens: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
        output_attentions: bool = False,
    ) -> dict:
        """Forward pass (LLaVA-style prepend).

        Sequence layout: [vision_tokens (N_v)] + [text_tokens (T)]
        Labels are shifted right by Llama's causal LM head internally;
        vision token positions always receive label=-100.

        Args:
            vision_tokens: (B, N_v, vision_dim) CONCHv1.5 patch embeddings
            input_ids: (B, T) tokenized instruction+response
            attention_mask: (B, T) 1=real, 0=pad
            labels: (B, T) -100 for instruction/pad positions
            output_attentions: whether to compute and return text→vision attention

        Returns:
            dict with keys:
                'loss': scalar LM loss (None if labels=None)
                'logits': (B, N_v+T, vocab_size)
                'text_to_vision_attn': (B, T, N_v) — only when output_attentions=True
                'n_vision_tokens': N_v (int)
        """
        assert self.llm is not None, "Call .load() before forward()"
        B, N_v, _ = vision_tokens.shape

        # Project vision tokens to LLM embedding space
        vision_embeds = self.adapter(vision_tokens)  # (B, N_v, llm_dim)

        # Embed text tokens using the LLM's embedding table
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, T, llm_dim)

        # Concatenate: vision first, then text
        inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)  # (B, N_v+T, llm_dim)

        # Attention mask: vision tokens are always attended to
        vis_mask = torch.ones(B, N_v, dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([vis_mask, attention_mask], dim=1)  # (B, N_v+T)

        # Labels: -100 for all vision token positions
        if labels is not None:
            vis_labels = torch.full((B, N_v), -100, dtype=labels.dtype, device=labels.device)
            full_labels = torch.cat([vis_labels, labels], dim=1)  # (B, N_v+T)
        else:
            full_labels = None

        llm_out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
            output_attentions=output_attentions,
            return_dict=True,
        )

        result: dict = {
            "loss": llm_out.loss,
            "logits": llm_out.logits,
            "n_vision_tokens": N_v,
        }

        if output_attentions and llm_out.attentions is not None:
            # attentions[i]: (B, num_heads, N_v+T, N_v+T)
            layer_attn = llm_out.attentions[self.attn_layer]
            # Text positions attending to vision positions
            text_to_vision = layer_attn[:, :, N_v:, :N_v]  # (B, H, T, N_v)
            result["text_to_vision_attn"] = text_to_vision.mean(dim=1)  # avg heads → (B, T, N_v)

        return result

    # ── Inference ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        vision_tokens: Tensor,
        prompt_input_ids: Tensor,
        max_new_tokens: int = 512,
        **generate_kwargs,
    ) -> Tensor:
        """Autoregressive generation conditioned on vision tokens.

        Args:
            vision_tokens: (1, N_v, vision_dim) — single slide
            prompt_input_ids: (1, T_prompt) tokenized instruction
            max_new_tokens: generation budget

        Returns:
            (1, T_out) generated token ids
        """
        assert self.llm is not None, "Call .load() before generate()"
        vision_embeds = self.adapter(vision_tokens)
        text_embeds = self.llm.get_input_embeddings()(prompt_input_ids)
        inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

    # ── Checkpoint I/O ───────────────────────────────────────────────────────

    def save_pretrained(self, checkpoint_dir: str) -> None:
        """Save LoRA adapter + linear adapter to checkpoint_dir.

        Args:
            checkpoint_dir: directory to write weights into
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        assert self.llm is not None, "Nothing to save — call .load() first"
        self.llm.save_pretrained(checkpoint_dir)  # saves LoRA adapter only
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(checkpoint_dir)
        torch.save(self.adapter.state_dict(), os.path.join(checkpoint_dir, "adapter.pt"))
        torch.save(self._cfg, os.path.join(checkpoint_dir, "vlm_config.pt"))
        logger.info("Checkpoint saved to %s", checkpoint_dir)

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str) -> GroundedVLM:
        """Load a GroundedVLM from a checkpoint saved by save_pretrained.

        Args:
            checkpoint_dir: directory previously written by save_pretrained

        Returns:
            loaded GroundedVLM (llm is a merged PEFT model, ready for inference)
        """
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = torch.load(os.path.join(checkpoint_dir, "vlm_config.pt"), map_location="cpu")
        model = cls(**cfg)

        logger.info("Loading base LLM from %s", cfg["llm_repo"])
        base_llm = AutoModelForCausalLM.from_pretrained(cfg["llm_repo"], torch_dtype=torch.bfloat16)
        model.llm = PeftModel.from_pretrained(base_llm, checkpoint_dir)
        model.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model.adapter.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "adapter.pt"), map_location="cpu")
        )
        return model
