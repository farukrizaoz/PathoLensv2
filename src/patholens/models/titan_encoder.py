"""TITAN slide encoder wrapper.

TITAN: Multimodal Whole Slide Foundation Model (MahmoodLab, 2024).
HF: MahmoodLab/TITAN (gated, requires institutional email).

Loads CONCHv1.5 patch encoder + TITAN slide encoder together.
Both frozen.

To be implemented in Hafta 2 — verify exact API in HF model card.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TITANEncoderWrapper(nn.Module):
    """Wraps TITAN's slide encoder with a clean forward API.

    Forward:
        Input: patch_embeddings (N_patches, 768), coordinates (N_patches, 2)
        Output: slide_tokens (N_slide_tokens, 768)
    """

    def __init__(self, hf_repo: str = "MahmoodLab/TITAN") -> None:
        super().__init__()
        # TODO: load TITAN with trust_remote_code=True
        # from transformers import AutoModel
        # self.titan = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
        # self.conch, self.conch_transform = self.titan.return_conch()
        raise NotImplementedError("Implement in Hafta 2")

    @torch.no_grad()
    def encode_patches(self, patches: Tensor) -> Tensor:
        """CONCHv1.5 forward: (B, 3, 448, 448) → (B, 768)."""
        raise NotImplementedError

    @torch.no_grad()
    def encode_slide(self, patch_embeddings: Tensor, coordinates: Tensor) -> Tensor:
        """TITAN slide encoder: (N_patches, 768) → (N_slide_tokens, 768)."""
        raise NotImplementedError
