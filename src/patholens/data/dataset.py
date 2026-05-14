"""PyTorch Dataset wrappers for SlideInstruction + precomputed embeddings.

Two dataset classes:
    SlideInstructionDataset  — caption-only baseline (no grounding)
    GroundingSlideDataset    — grounding-aware training (v15 patch tokens + per-sentence GT)

Design note: GroundingSlideDataset uses CONCHv1.5 patch embeddings (768-d) as vision tokens,
NOT TITAN slide tokens.  This keeps the vision token space 1:1 with the grounding GT (patch-
level distributions from grounding_targets.py).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    """Split a pathology report paragraph into individual sentences.

    Uses punctuation boundary splitting; keeps each non-empty sentence.

    Args:
        text: free-form text (typically one paragraph or structured report)

    Returns:
        list of stripped sentences
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def _build_chat_tokens(
    tokenizer: Any,
    instruction: str,
    response: str,
    max_text_tokens: int,
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Tokenize instruction + response in Llama-3.2 chat format.

    Sequence: <|begin_of_text|>[user header + instruction + eot][assistant header]
              [response tokens][eos]

    Labels: -100 for the entire instruction / header prefix; real token ids for
            the response portion (causal LM target).

    Args:
        tokenizer: loaded HuggingFace AutoTokenizer (Llama-3.2-3B-Instruct)
        instruction: the user-turn text (e.g. "Generate a pathology report …")
        response: the assistant-turn text (the structured report)
        max_text_tokens: maximum total token length (truncated if longer)

    Returns:
        input_ids: (T,) long tensor
        attention_mask: (T,) long tensor
        labels: (T,) long tensor — -100 for instruction positions
        response_start: index in input_ids where the response begins (for span detection)
    """
    messages = [{"role": "user", "content": instruction}]
    prompt_only: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize prompt alone to know where response starts
    prompt_ids: list[int] = tokenizer(
        prompt_only,
        add_special_tokens=False,
    ).input_ids

    full_text = prompt_only + response + tokenizer.eos_token
    full_enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_text_tokens,
        add_special_tokens=False,
    )
    input_ids: Tensor = full_enc.input_ids[0]
    attention_mask: Tensor = full_enc.attention_mask[0]

    response_start = min(len(prompt_ids), len(input_ids))

    labels = input_ids.clone()
    labels[:response_start] = -100  # mask instruction / header positions

    return input_ids, attention_mask, labels, response_start


def _find_sentence_spans(
    tokenizer: Any,
    sentences: list[str],
    input_ids: Tensor,
    response_start: int,
) -> list[tuple[int, int]]:
    """Approximate token span for each sentence within the response part of input_ids.

    Method: tokenize each sentence without special tokens, accumulate lengths from
    response_start.  Spans may be slightly off at subword boundaries but are good
    enough to aggregate attention over the right region.

    Args:
        tokenizer: loaded HuggingFace tokenizer
        sentences: response split into sentences
        input_ids: (T,) full tokenized sequence
        response_start: token index where response begins in input_ids

    Returns:
        list of (start, end) pairs — indices into input_ids (end is exclusive)
    """
    spans: list[tuple[int, int]] = []
    cursor = response_start
    total = len(input_ids)

    for sentence in sentences:
        sent_ids = tokenizer(sentence, add_special_tokens=False).input_ids
        end = min(cursor + len(sent_ids), total)
        spans.append((cursor, end))
        cursor = end
        if cursor >= total:
            break

    # Fill remaining sentences with a zero-width span at the end
    while len(spans) < len(sentences):
        spans.append((total - 1, total - 1))

    return spans


# ─── Datasets ─────────────────────────────────────────────────────────────────


class SlideInstructionDataset(Dataset):
    """Dataset that yields (patch_embeddings_v15, caption) pairs for caption-only training.

    Loads CONCHv1.5 patch embeddings (patch_embeddings_v15, 768-d) as vision tokens.
    Skips slides whose HDF5 file is missing.

    Args:
        instruction_json: path to slideinstruction/{train,val,test}.json
        embeddings_dir: path to data/processed/embeddings/tcga_brca/
        max_vision_tokens: cap on patch token count (truncate if more)
    """

    def __init__(
        self,
        instruction_json: str | Path,
        embeddings_dir: str | Path,
        max_vision_tokens: int = 1024,
    ) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        self.max_vision_tokens = max_vision_tokens

        with open(instruction_json) as f:
            all_examples: list[dict] = json.load(f)

        self.examples = [
            e for e in all_examples if (self.embeddings_dir / f"{e['slide_id']}.h5").exists()
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        h5_path = self.embeddings_dir / f"{ex['slide_id']}.h5"

        with h5py.File(h5_path, "r") as f:
            patch_embeddings = torch.from_numpy(f["patch_embeddings_v15"][:]).float()

        if patch_embeddings.shape[0] > self.max_vision_tokens:
            patch_embeddings = patch_embeddings[: self.max_vision_tokens]

        return {
            "slide_id": ex["slide_id"],
            "patch_embeddings_v15": patch_embeddings,
            "caption": ex["caption"],
            "vqa_pairs": ex.get("vqa_pairs", []),
        }


class GroundingSlideDataset(Dataset):
    """Dataset for grounding-aware training.

    Each item provides:
        - CONCHv1.5 patch embeddings as vision tokens (capped at max_vision_tokens)
        - CONCH v1 patch embeddings and coordinates for pseudo-GT computation
        - Tokenized instruction + response in Llama-3.2 chat format
        - Response split into sentences with approximate token spans
        - Path to pre-saved BCSS patch mask HDF5 (or None if absent)

    Args:
        instruction_json: path to slideinstruction/{train,val,test}.json
        embeddings_dir: path to data/processed/embeddings/tcga_brca/
        tokenizer: loaded HuggingFace AutoTokenizer for chat template + span detection
        bcss_masks_dir: path to data/processed/bcss_patch_masks/ (None = pseudo-GT only)
        pseudo_gt_cache_dir: root directory for PseudoGTCache (passed through to trainer)
        max_vision_tokens: maximum number of patch tokens fed to LLM as vision prefix
        max_text_tokens: maximum token count for the instruction + response sequence
    """

    def __init__(
        self,
        instruction_json: str | Path,
        embeddings_dir: str | Path,
        tokenizer: Any,
        bcss_masks_dir: str | Path | None = None,
        pseudo_gt_cache_dir: str | Path | None = None,
        max_vision_tokens: int = 1024,
        max_text_tokens: int = 512,
    ) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        self.bcss_masks_dir = Path(bcss_masks_dir) if bcss_masks_dir else None
        self.pseudo_gt_cache_dir = str(pseudo_gt_cache_dir) if pseudo_gt_cache_dir else None
        self.tokenizer = tokenizer
        self.max_vision_tokens = max_vision_tokens
        self.max_text_tokens = max_text_tokens

        with open(instruction_json) as f:
            all_examples: list[dict] = json.load(f)

        self.examples = [
            e for e in all_examples if (self.embeddings_dir / f"{e['slide_id']}.h5").exists()
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        slide_id: str = ex["slide_id"]
        h5_path = self.embeddings_dir / f"{slide_id}.h5"

        with h5py.File(h5_path, "r") as f:
            patch_embeddings_v15 = torch.from_numpy(f["patch_embeddings_v15"][:]).float()
            patch_embeddings_v1: np.ndarray = f["patch_embeddings_v1"][:]
            coordinates: np.ndarray = f["coordinates"][:]

        # Truncate all patch arrays to max_vision_tokens consistently
        n = patch_embeddings_v15.shape[0]
        if n > self.max_vision_tokens:
            patch_embeddings_v15 = patch_embeddings_v15[: self.max_vision_tokens]
            patch_embeddings_v1 = patch_embeddings_v1[: self.max_vision_tokens]
            coordinates = coordinates[: self.max_vision_tokens]

        caption: str = ex["caption"]
        sentences = _split_sentences(caption)
        if not sentences:
            sentences = [caption]

        instruction: str = ex.get(
            "instruction",
            "Generate a structured pathology report for this whole-slide image.",
        )

        input_ids, attention_mask, labels, response_start = _build_chat_tokens(
            self.tokenizer, instruction, caption, self.max_text_tokens
        )

        sentence_spans = _find_sentence_spans(self.tokenizer, sentences, input_ids, response_start)

        bcss_h5_path: str | None = None
        if self.bcss_masks_dir is not None:
            candidate = self.bcss_masks_dir / f"{slide_id}.h5"
            if candidate.exists():
                bcss_h5_path = str(candidate)

        return {
            "slide_id": slide_id,
            "patch_embeddings_v15": patch_embeddings_v15,  # (N_v, 768) Tensor
            "patch_embeddings_v1": patch_embeddings_v1,  # (N_v, 512) numpy
            "coordinates": coordinates,  # (N_v, 2) numpy int32
            "input_ids": input_ids,  # (T,) long
            "attention_mask": attention_mask,  # (T,) long
            "labels": labels,  # (T,) long
            "sentences": sentences,  # list[str]
            "sentence_spans": sentence_spans,  # list[tuple[int, int]]
            "bcss_h5_path": bcss_h5_path,  # str | None
            "pseudo_gt_cache_dir": self.pseudo_gt_cache_dir,  # str | None
        }


# ─── Collate ──────────────────────────────────────────────────────────────────


def grounding_collate_fn(batch: list[dict]) -> dict:
    """Collate function for GroundingSlideDataset.

    Pads input_ids / attention_mask / labels to the longest sequence in the batch.
    Variable-length patch arrays and metadata are kept as lists.

    Args:
        batch: list of dicts from GroundingSlideDataset.__getitem__

    Returns:
        collated batch dict with padded tensors and lists for variable-size fields
    """
    input_ids_list = [item["input_ids"] for item in batch]
    attn_masks = [item["attention_mask"] for item in batch]
    labels_list = [item["labels"] for item in batch]

    max_len = max(t.shape[0] for t in input_ids_list)

    def _pad1d(tensors: list[Tensor], pad_value: int) -> Tensor:
        out = torch.full((len(tensors), max_len), pad_value, dtype=torch.long)
        for i, t in enumerate(tensors):
            out[i, : t.shape[0]] = t
        return out

    return {
        "slide_ids": [item["slide_id"] for item in batch],
        "patch_embeddings_v15": [item["patch_embeddings_v15"] for item in batch],
        "patch_embeddings_v1": [item["patch_embeddings_v1"] for item in batch],
        "coordinates": [item["coordinates"] for item in batch],
        "input_ids": _pad1d(input_ids_list, pad_value=0),
        "attention_mask": _pad1d(attn_masks, pad_value=0),
        "labels": _pad1d(labels_list, pad_value=-100),
        "sentences": [item["sentences"] for item in batch],
        "sentence_spans": [item["sentence_spans"] for item in batch],
        "bcss_h5_paths": [item["bcss_h5_path"] for item in batch],
        "pseudo_gt_cache_dirs": [item["pseudo_gt_cache_dir"] for item in batch],
    }
