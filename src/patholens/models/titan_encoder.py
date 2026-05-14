"""TITAN slide encoder wrapper (MahmoodLab/TITAN).

TITAN aggregates CONCHv1.5 patch embeddings + coordinates into slide-level
tokens via a hierarchical transformer. Both TITAN and the bundled CONCHv1.5
are frozen during training.

HuggingFace: MahmoodLab/TITAN (gated, institutional email required).
Paper: https://arxiv.org/abs/2411.19666
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

HF_TITAN = "MahmoodLab/TITAN"


class TITANEncoderWrapper(nn.Module):
    """Frozen TITAN slide encoder with a clean forward API.

    Loads the TITAN AutoModel (trust_remote_code=True) which bundles CONCHv1.5.
    Both models run in eval mode with no gradient tracking.

    Args:
        hf_repo: HuggingFace repository ID for TITAN
        device: torch device string ("cuda", "cpu")
        dtype: compute dtype; bf16 recommended on A6000
    """

    def __init__(
        self,
        hf_repo: str = HF_TITAN,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hf_repo = hf_repo
        self.device_str = device
        self.dtype = dtype
        self._titan: object | None = None

    def load(self) -> TITANEncoderWrapper:
        """Load TITAN from HuggingFace and move to device.

        Must be called before encode_patches / encode_slide.
        Requires HF_TOKEN or prior huggingface-cli login.

        Returns:
            self (for chaining)
        """
        from transformers import AutoModel

        titan = AutoModel.from_pretrained(
            self.hf_repo,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )
        titan = titan.to(self.device_str).eval()
        # Freeze all parameters
        for param in titan.parameters():
            param.requires_grad_(False)

        self._titan = titan
        return self

    @torch.inference_mode()
    def encode_patches(self, patches: Tensor) -> Tensor:
        """Run CONCHv1.5 patch encoder bundled inside TITAN.

        Args:
            patches: (B, 3, 448, 448) float tensor on any device

        Returns:
            (B, 768) float32 patch embeddings
        """
        assert self._titan is not None, "Call .load() first"
        conch, _ = self._titan.return_conch()
        patches = patches.to(self.device_str, dtype=self.dtype)
        emb = conch.encode_image(patches)
        return emb.float()

    @torch.inference_mode()
    def encode_slide(self, patch_embeddings: Tensor, coordinates: Tensor) -> Tensor:
        """Run TITAN slide encoder to produce slide-level tokens.

        Args:
            patch_embeddings: (N_patches, 768) float tensor
            coordinates: (N_patches, 2) int tensor — (x, y) origin in level-0 coords

        Returns:
            (N_slide_tokens, 768) float32 slide token sequence

        Note:
            Verify exact API on RunPod with the TITAN model card:
            https://huggingface.co/MahmoodLab/TITAN
            The encode_slide_from_patch_features method expects batched tensors.
        """
        assert self._titan is not None, "Call .load() first"
        # TITAN expects a batch dimension
        patch_embeddings = patch_embeddings.unsqueeze(0).to(self.device_str, dtype=self.dtype)
        coordinates = coordinates.unsqueeze(0).to(self.device_str)

        output = self._titan.encode_slide_from_patch_features(
            patch_features=patch_embeddings,
            coords=coordinates,
        )

        # output is typically a dict with 'last_hidden_state' or a tuple
        if isinstance(output, dict):
            slide_tokens = output["last_hidden_state"]
        elif isinstance(output, tuple | list):
            slide_tokens = output[0]
        else:
            slide_tokens = output

        # Remove batch dimension → (N_slide_tokens, 768)
        if slide_tokens.ndim == 3:
            slide_tokens = slide_tokens.squeeze(0)

        return slide_tokens.float()

    @property
    def conch_transform(self) -> object:
        """Return the CONCHv1.5 image preprocessing transform."""
        assert self._titan is not None, "Call .load() first"
        _, transform = self._titan.return_conch()
        return transform
