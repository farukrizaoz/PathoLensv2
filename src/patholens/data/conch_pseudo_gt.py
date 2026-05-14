"""CONCH v1 pseudo-GT generator for grounding supervision.

For each report sentence, computes a soft attention distribution over slide
patches by taking cosine similarity between the CONCH v1 text embedding and
the CONCH v1 patch embeddings stored in the HDF5 cache.

The resulting distribution serves as the supervision target for GroundingLoss
when no explicit pixel-level annotation (BCSS) covers the patch.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DEFAULT_TEMPERATURE = 0.1
CACHE_SUBDIR = "pseudo_gt_cache"


def compute_pseudo_gt(
    sentence: str,
    patch_embeddings_v1: NDArray,
    conch_v1_encoder: object,
    temperature: float = DEFAULT_TEMPERATURE,
) -> NDArray:
    """Compute a softmax attention distribution over patches for a single sentence.

    Args:
        sentence: a single sentence from a pathology report
        patch_embeddings_v1: (N_patches, 512) float array — CONCH v1 vision embeddings
        conch_v1_encoder: loaded ConchV1Encoder instance
        temperature: softmax sharpening temperature (lower = sharper)

    Returns:
        (N_patches,) float32 softmax distribution over patches
    """
    text_emb = conch_v1_encoder.encode_text(sentence)  # (1, 512)
    text_emb_np = text_emb.cpu().numpy()  # (1, 512)

    # Cosine similarity: both L2-normalized, so dot product = cosine sim
    patch_emb_normed = patch_embeddings_v1 / (
        np.linalg.norm(patch_embeddings_v1, axis=-1, keepdims=True) + 1e-8
    )
    similarity = patch_emb_normed @ text_emb_np.T  # (N_patches, 1)
    similarity = similarity[:, 0]  # (N_patches,)

    logits = similarity / temperature
    # Numerically stable softmax
    logits -= logits.max()
    exp_logits = np.exp(logits)
    distribution = exp_logits / exp_logits.sum()
    return distribution.astype(np.float32)


def compute_pseudo_gt_batch(
    sentences: list[str],
    patch_embeddings_v1: NDArray,
    conch_v1_encoder: object,
    temperature: float = DEFAULT_TEMPERATURE,
) -> NDArray:
    """Compute pseudo-GT distributions for multiple sentences at once.

    Args:
        sentences: list of S sentences
        patch_embeddings_v1: (N_patches, 512) float array
        conch_v1_encoder: loaded ConchV1Encoder instance
        temperature: softmax temperature

    Returns:
        (S, N_patches) float32 array — one distribution per sentence
    """
    return np.stack(
        [
            compute_pseudo_gt(s, patch_embeddings_v1, conch_v1_encoder, temperature)
            for s in sentences
        ],
        axis=0,
    )


class PseudoGTCache:
    """Disk-backed cache for pseudo-GT distributions.

    Avoids recomputing text embeddings on every epoch. Cache is keyed by
    (slide_id, sentence_hash) and stored as a JSON-indexed numpy archive.

    Args:
        cache_dir: directory to write cache files under CACHE_SUBDIR/
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir) / CACHE_SUBDIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, slide_id: str) -> Path:
        return self.cache_dir / f"{slide_id}.npz"

    def _sentence_key(self, sentence: str) -> str:
        return hashlib.md5(sentence.encode()).hexdigest()[:16]

    def get(self, slide_id: str, sentence: str) -> NDArray | None:
        """Return cached distribution if available, else None."""
        path = self._cache_path(slide_id)
        if not path.exists():
            return None
        key = self._sentence_key(sentence)
        data = np.load(path, allow_pickle=True)
        if key in data:
            return data[key]
        return None

    def put(self, slide_id: str, sentence: str, distribution: NDArray) -> None:
        """Write a distribution to the cache, merging with existing entries."""
        path = self._cache_path(slide_id)
        existing: dict[str, NDArray] = {}
        if path.exists():
            loaded = np.load(path, allow_pickle=True)
            existing = dict(loaded)
        key = self._sentence_key(sentence)
        existing[key] = distribution
        np.savez_compressed(path, **existing)

    def get_or_compute(
        self,
        slide_id: str,
        sentence: str,
        patch_embeddings_v1: NDArray,
        conch_v1_encoder: object,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> NDArray:
        """Return cached distribution, computing and storing it if missing.

        Args:
            slide_id: TCGA slide identifier
            sentence: a single report sentence
            patch_embeddings_v1: (N_patches, 512) CONCH v1 patch embeddings
            conch_v1_encoder: loaded ConchV1Encoder instance
            temperature: softmax temperature

        Returns:
            (N_patches,) float32 distribution
        """
        dist = self.get(slide_id, sentence)
        if dist is not None:
            return dist
        dist = compute_pseudo_gt(sentence, patch_embeddings_v1, conch_v1_encoder, temperature)
        self.put(slide_id, sentence, dist)
        return dist
