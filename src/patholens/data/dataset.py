"""PyTorch Dataset wrappers for SlideInstruction + precomputed embeddings.

To be implemented in Hafta 2.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class SlideInstructionDataset(Dataset):
    """Dataset that yields (slide_tokens, instruction, response) tuples.

    Args:
        instruction_json: path to slideinstruction/{train,val,test}.json
        embeddings_dir: path to data/processed/embeddings/tcga_brca/
        max_vision_tokens: cap on slide_tokens (truncate or pool if more)
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
            self.examples = json.load(f)

        # TODO: filter examples whose embedding HDF5 exists
        # self.examples = [e for e in self.examples if (self.embeddings_dir / f"{e['slide_id']}.h5").exists()]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        h5_path = self.embeddings_dir / f"{ex['slide_id']}.h5"

        with h5py.File(h5_path, "r") as f:
            slide_tokens = torch.from_numpy(f["slide_tokens"][:]).float()

        # Truncate if too many tokens
        if slide_tokens.shape[0] > self.max_vision_tokens:
            slide_tokens = slide_tokens[: self.max_vision_tokens]

        return {
            "slide_id": ex["slide_id"],
            "slide_tokens": slide_tokens,
            "caption": ex["caption"],
            "vqa_pairs": ex.get("vqa_pairs", []),
        }
