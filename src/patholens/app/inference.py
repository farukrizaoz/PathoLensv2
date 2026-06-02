"""Inference helper: generate a report AND return per-sentence attention.

`GroundedVLM.generate()` returns token ids only. The grounding-aware demo
needs the text→vision attention sliced per generated sentence. We achieve
that by running generate first, then a single forward pass on
[prompt || generated] with `output_attentions=True` to extract the
text→vision attention block, then aggregating over each sentence's
response-token span.

Reuses `_split_sentences` and `_find_sentence_spans` from
`patholens.data.dataset` so the spans are identified the same way the
trainer does.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from patholens.data.dataset import _find_sentence_spans, _split_sentences

if TYPE_CHECKING:
    from patholens.models.grounded_vlm import GroundedVLM

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = (
    "Generate a structured pathology report for this whole-slide image."
)


def _build_prompt_ids(tokenizer, instruction: str, device: torch.device) -> Tensor:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


@torch.inference_mode()
def generate_with_attention(
    model: "GroundedVLM",
    vision_tokens: Tensor,
    instruction: str = DEFAULT_INSTRUCTION,
    max_new_tokens: int = 384,
) -> dict:
    """Generate a report and return per-sentence text→vision attention.

    Args:
        model: a GroundedVLM loaded via `from_pretrained` (adapter in bf16).
        vision_tokens: (1, N_v, vision_dim) bf16 on the model device.
        instruction: user-turn text used to prompt the LLM.
        max_new_tokens: generation cap.

    Returns:
        dict with keys:
            text                — full generated report (str)
            sentences           — list[str]
            per_sentence_attn   — list[np.ndarray], each shape (N_v,), sums to 1
            n_vision_tokens     — int
            instruction         — echo of the instruction
    """
    assert model.llm is not None and model.tokenizer is not None, "model not loaded"
    tokenizer = model.tokenizer
    device = next(model.adapter.parameters()).device
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_v = vision_tokens.shape[1]

    # ── 1. Generate
    prompt_ids = _build_prompt_ids(tokenizer, instruction, device)
    gen_ids = model.generate(
        vision_tokens=vision_tokens,
        prompt_input_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )  # (1, T_new) — only new tokens when called with inputs_embeds
    new_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
    logger.info("Generated %d new tokens", gen_ids.shape[1])

    # ── 2. Re-tokenize [prompt || response] so we have token positions for both
    full_ids = torch.cat([prompt_ids, gen_ids], dim=1)  # (1, T_total)
    attention_mask = torch.ones_like(full_ids)
    response_start = prompt_ids.shape[1]

    # ── 3. Forward with attentions to grab text→vision block
    out = model(
        vision_tokens=vision_tokens,
        input_ids=full_ids,
        attention_mask=attention_mask,
        labels=None,
        output_attentions=True,
    )
    text_to_vision = out["text_to_vision_attn"]  # (1, T_total, N_v)
    if text_to_vision is None:
        raise RuntimeError("Model did not return text_to_vision_attn — check attn_layer config")

    # ── 4. Split sentences and locate spans in the response portion
    sentences = _split_sentences(new_text)
    if not sentences:
        sentences = [new_text] if new_text else ["(empty generation)"]
    spans = _find_sentence_spans(tokenizer, sentences, full_ids[0], response_start)

    # ── 5. Aggregate attention over each span
    per_sentence: list[np.ndarray] = []
    attn_cpu = text_to_vision[0].float().cpu().numpy()  # (T_total, N_v)
    for start, end in spans:
        if end <= start:
            per_sentence.append(np.full(n_v, 1.0 / n_v, dtype=np.float32))
            continue
        sent_attn = attn_cpu[start:end].mean(axis=0)  # (N_v,)
        s = float(sent_attn.sum())
        per_sentence.append((sent_attn / s).astype(np.float32) if s > 1e-8 else
                            np.full(n_v, 1.0 / n_v, dtype=np.float32))

    return {
        "text": new_text,
        "sentences": sentences,
        "per_sentence_attn": per_sentence,
        "n_vision_tokens": n_v,
        "instruction": instruction,
    }
