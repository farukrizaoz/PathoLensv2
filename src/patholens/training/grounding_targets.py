"""Hybrid grounding target builder.

For each (slide, sentence) pair, routes to the appropriate supervision signal:

    BCSS mask  →  when roi_coverage[i] is True AND the sentence mentions a
                  class concept that has a non-trivial BCSS mask on that patch
    CONCH pseudo-GT  →  everywhere else (most patches and all non-concept sentences)

The resulting (N_patches,) float32 distribution is used as `gt_attn` in
GroundingLoss.  Entries are re-normalised to sum to 1 after masking.

Concept detection is intentionally simple: substring matching against a small
keyword dictionary.  A sentence like "lymphocytic infiltrate at the periphery"
maps to BCSS mask_lymph.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from patholens.data.conch_pseudo_gt import PseudoGTCache

logger = logging.getLogger(__name__)

# ─── Concept → BCSS channel mapping ─────────────────────────────────────────

CONCEPT_TO_BCSS_KEY: dict[str, str] = {
    "tumor": "mask_tumor",
    "carcinoma": "mask_tumor",
    "invasive": "mask_tumor",
    "malignant": "mask_tumor",
    "idc": "mask_tumor",
    "ilc": "mask_tumor",
    "stroma": "mask_stroma",
    "stromal": "mask_stroma",
    "desmoplastic": "mask_stroma",
    "fibrosis": "mask_stroma",
    "lymph": "mask_lymph",
    "lymphocytic": "mask_lymph",
    "inflammatory": "mask_lymph",
    "til": "mask_lymph",
    "necrosis": "mask_necrosis",
    "necrotic": "mask_necrosis",
    "debris": "mask_necrosis",
    "karyorrhectic": "mask_necrosis",
}

# Minimum fraction in BCSS mask to trust it as GT for a patch
BCSS_MASK_THRESHOLD = 0.30

# Minimum fraction of patches covered by BCSS in a distribution to trust it
# over pseudo-GT (avoid distributions where only 1–2 patches have BCSS data)
MIN_BCSS_COVERAGE_FRACTION = 0.05


def _detect_concept(sentence: str) -> str | None:
    """Return the BCSS channel key for a sentence, or None if no concept matches.

    Matching is case-insensitive substring search.  Returns the first match.

    Args:
        sentence: a single report sentence

    Returns:
        One of 'mask_tumor', 'mask_stroma', 'mask_lymph', 'mask_necrosis', or None
    """
    lower = sentence.lower()
    for keyword, channel in CONCEPT_TO_BCSS_KEY.items():
        if keyword in lower:
            return channel
    return None


def _normalise(arr: NDArray) -> NDArray:
    """Normalise arr to sum to 1; return uniform if all zeros."""
    total = float(arr.sum())
    if total < 1e-8:
        return np.full_like(arr, 1.0 / len(arr))
    return (arr / total).astype(np.float32)


def build_target(
    slide_id: str,
    sentence: str,
    patch_embeddings_v1: NDArray,
    coordinates: NDArray,
    conch_v1_encoder: object,
    bcss_masks: dict[str, NDArray] | None,
    pseudo_gt_cache: PseudoGTCache,
    temperature: float = 0.1,
    grounding_source: Literal["conch_only", "bcss_hybrid"] = "bcss_hybrid",
) -> tuple[NDArray, bool]:
    """Build a (N_patches,) grounding target distribution for one sentence.

    Routing logic (when grounding_source='bcss_hybrid'):
        1. Detect concept keyword → BCSS channel
        2. If BCSS masks exist for this slide AND roi_coverage is non-trivial
           AND the channel mask passes the threshold → use BCSS mask as target
        3. Otherwise fall back to CONCH v1 pseudo-GT

    Args:
        slide_id: TCGA slide identifier
        sentence: a single pathology report sentence
        patch_embeddings_v1: (N_patches, 512) CONCH v1 patch embeddings
        coordinates: (N_patches, 2) patch origins — used only for pseudo-GT cache key
        conch_v1_encoder: loaded ConchV1Encoder instance
        bcss_masks: dict returned by BcssLoader.build_patch_masks(), or None
        pseudo_gt_cache: PseudoGTCache instance (handles caching)
        temperature: softmax temperature for pseudo-GT
        grounding_source: 'conch_only' forces pseudo-GT; 'bcss_hybrid' enables BCSS routing

    Returns:
        target: (N_patches,) float32 distribution summing to 1
        used_bcss: True if BCSS mask was used (False → pseudo-GT)
    """
    n_patches = len(patch_embeddings_v1)

    # ── Pseudo-GT (always available as fallback) ─────────────────────────────
    pseudo_gt = pseudo_gt_cache.get_or_compute(
        slide_id=slide_id,
        sentence=sentence,
        patch_embeddings_v1=patch_embeddings_v1,
        conch_v1_encoder=conch_v1_encoder,
        temperature=temperature,
    )

    if grounding_source == "conch_only" or bcss_masks is None:
        return pseudo_gt, False

    # ── BCSS routing ─────────────────────────────────────────────────────────
    concept_channel = _detect_concept(sentence)
    if concept_channel is None:
        return pseudo_gt, False

    mask_values: NDArray = bcss_masks.get(concept_channel)  # (N_patches,) float32
    roi_coverage: NDArray = bcss_masks.get("roi_coverage")  # (N_patches,) bool

    if mask_values is None or roi_coverage is None:
        logger.debug(
            "BCSS channel %s not found for %s — using pseudo-GT", concept_channel, slide_id
        )
        return pseudo_gt, False

    # How many patches are inside the BCSS ROI?
    roi_fraction = float(roi_coverage.sum()) / max(n_patches, 1)
    if roi_fraction < MIN_BCSS_COVERAGE_FRACTION:
        return pseudo_gt, False  # too few annotated patches — trust pseudo-GT

    # Build a target from BCSS mask: use the mask value directly as unnormalised weight
    bcss_target = np.where(
        (roi_coverage) & (mask_values >= BCSS_MASK_THRESHOLD),
        mask_values,
        0.0,
    ).astype(np.float32)

    # If almost no patch passes the threshold, fall back to pseudo-GT
    if bcss_target.sum() < 1e-8:
        return pseudo_gt, False

    return _normalise(bcss_target), True
