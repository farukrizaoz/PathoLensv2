"""CONCH vision encoders: v1 (vision+text, 512-d) and v1.5 (vision-only, 768-d).

Both models are gated on HuggingFace (MahmoodLab). Requires:
    - HF_TOKEN in environment or huggingface-cli login
    - Institutional email approval

Usage:
    encoder = ConchEncoder.load_v1()   # pseudo-GT text+vision
    encoder = ConchEncoder.load_v15()  # TITAN patch input (vision only)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

HF_CONCH_V1 = "MahmoodLab/conch"
HF_CONCH_V15 = "MahmoodLab/conchv1_5"

# PubMedBERT tokenizer used by CONCH v1 text encoder
_PUBMEDBERT_REPO = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# CONCH v1 text encoder maximum token length (model constraint)
_CONCH_V1_MAX_TEXT_LEN = 128


class ConchV1Encoder:
    """CONCH v1 vision+text encoder wrapper (512-d shared space).

    Used for pseudo-GT generation: cosine similarity between text sentence
    embedding and patch vision embeddings.

    Args:
        device: torch device string ("cuda", "cpu", etc.)
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self._model: object | None = None
        self._tokenizer: object | None = None
        self._transform: object | None = None

    def load(self) -> ConchV1Encoder:
        """Load CONCH v1 model and tokenizer from HuggingFace."""
        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except ImportError as exc:
            raise ImportError(
                "Install CONCH: pip install git+https://github.com/mahmoodlab/CONCH.git"
            ) from exc

        from transformers import AutoTokenizer

        model, transform = create_model_from_pretrained(
            f"hf-hub:{HF_CONCH_V1}", force_quick_gelu=True
        )
        model = model.to(self.device).eval()

        tokenizer = AutoTokenizer.from_pretrained(_PUBMEDBERT_REPO)

        self._model = model
        self._transform = transform
        self._tokenizer = tokenizer
        return self

    @property
    def transform(self) -> object:
        """Return the CONCH v1 image preprocessing transform."""
        assert self._transform is not None, "Call .load() first"
        return self._transform

    @torch.inference_mode()
    def encode_patches(self, patches: Tensor) -> Tensor:
        """Encode image patches to 512-d vision embeddings.

        Args:
            patches: (B, 3, 448, 448) float32 tensor, values in [0, 1]

        Returns:
            (B, 512) L2-normalized embeddings
        """
        assert self._model is not None, "Call .load() first"
        emb = self._model.encode_image(patches.to(self.device))
        return F.normalize(emb.float(), dim=-1)

    @torch.inference_mode()
    def encode_text(self, sentence: str) -> Tensor:
        """Encode a text string to a 512-d text embedding.

        Args:
            sentence: a single pathology report sentence

        Returns:
            (1, 512) L2-normalized embedding
        """
        assert self._model is not None and self._tokenizer is not None, "Call .load() first"
        tokenizer = self._tokenizer

        encoded = tokenizer(
            [sentence],
            max_length=_CONCH_V1_MAX_TEXT_LEN - 1,  # leave room for pad
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # CONCH v1 expects exactly 128 tokens including a trailing pad
        input_ids = F.pad(
            encoded["input_ids"],
            (0, 1),
            value=tokenizer.pad_token_id,
        ).to(self.device)

        emb = self._model.encode_text(input_ids)
        return F.normalize(emb.float(), dim=-1)


class ConchV15Encoder:
    """CONCHv1.5 vision-only encoder wrapper (768-d).

    Used to produce patch embeddings for TITAN's slide encoder.
    Loaded through the TITAN model bundle.

    Args:
        device: torch device string
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self._model: object | None = None
        self._transform: object | None = None

    def load_from_titan(self, titan_model: object) -> ConchV15Encoder:
        """Extract the bundled CONCHv1.5 from an already-loaded TITAN model.

        Args:
            titan_model: loaded TITAN AutoModel instance

        Returns:
            self (for chaining)
        """
        conch, transform = titan_model.return_conch()
        self._model = conch.to(self.device).eval()
        self._transform = transform
        return self

    def load_standalone(self) -> ConchV15Encoder:
        """Load CONCHv1.5 independently from HuggingFace.

        Use this when TITAN is not loaded (e.g., patch-only inspection).
        """
        from transformers import AutoModel

        model = (
            AutoModel.from_pretrained(HF_CONCH_V15, trust_remote_code=True).to(self.device).eval()
        )
        self._model = model
        # CONCHv1.5 uses the same 448x448 normalization as CONCHv1
        import torchvision.transforms as T

        self._transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        return self

    @property
    def transform(self) -> object:
        """Return the preprocessing transform for this encoder."""
        assert self._transform is not None, "Call .load_from_titan() or .load_standalone() first"
        return self._transform

    @torch.inference_mode()
    def encode_patches(self, patches: Tensor) -> Tensor:
        """Encode image patches to 768-d vision embeddings.

        Args:
            patches: (B, 3, 448, 448) float32 tensor

        Returns:
            (B, 768) L2-normalized embeddings
        """
        assert self._model is not None, "Call .load_from_titan() or .load_standalone() first"
        emb = self._model.encode_image(patches.to(self.device))
        return F.normalize(emb.float(), dim=-1)
