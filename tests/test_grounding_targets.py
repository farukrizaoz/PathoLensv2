"""Tests for grounding_targets: routing logic and concept detection."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from patholens.data.conch_pseudo_gt import PseudoGTCache
from patholens.training.grounding_targets import (
    _detect_concept,
    _normalise,
    build_target,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

N_PATCHES = 50
DIM_V1 = 512


def _mock_cache(fixed_distribution: np.ndarray) -> PseudoGTCache:
    """PseudoGTCache that always returns a fixed distribution without disk access."""
    cache = MagicMock(spec=PseudoGTCache)
    cache.get_or_compute.return_value = fixed_distribution
    return cache


def _mock_encoder() -> object:
    encoder = MagicMock()
    encoder.encode_text.return_value = np.zeros((1, DIM_V1), dtype=np.float32)
    return encoder


def _uniform_pseudo_gt() -> np.ndarray:
    dist = np.ones(N_PATCHES, dtype=np.float32)
    return dist / dist.sum()


def _patch_embeddings() -> np.ndarray:
    emb = np.random.randn(N_PATCHES, DIM_V1).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=-1, keepdims=True)
    return emb


def _coordinates() -> np.ndarray:
    return np.zeros((N_PATCHES, 2), dtype=np.int32)


# ─── Concept detection ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("Invasive ductal carcinoma, grade 2", "mask_tumor"),
        ("Lymphocytic infiltrate at the periphery", "mask_lymph"),
        ("Desmoplastic stroma with fibrosis", "mask_stroma"),
        ("Focal necrosis with karyorrhectic debris", "mask_necrosis"),
        ("The tumor shows ILC pattern", "mask_tumor"),
        ("TIL score is 30%", "mask_lymph"),
        ("No significant pathology noted", None),
        ("Margins negative", None),
    ],
)
def test_detect_concept(sentence: str, expected: str | None) -> None:
    assert _detect_concept(sentence) == expected


# ─── Normalise ────────────────────────────────────────────────────────────────


def test_normalise_sums_to_one() -> None:
    arr = np.array([0.1, 0.4, 0.5], dtype=np.float32)
    result = _normalise(arr)
    assert abs(result.sum() - 1.0) < 1e-6


def test_normalise_all_zeros_returns_uniform() -> None:
    arr = np.zeros(10, dtype=np.float32)
    result = _normalise(arr)
    assert abs(result.sum() - 1.0) < 1e-6
    assert np.allclose(result, 0.1)


# ─── build_target: conch_only mode ───────────────────────────────────────────


def test_conch_only_always_returns_pseudo_gt() -> None:
    pseudo = _uniform_pseudo_gt()
    target, used_bcss = build_target(
        slide_id="TCGA-TEST",
        sentence="Invasive carcinoma with necrosis",
        patch_embeddings_v1=_patch_embeddings(),
        coordinates=_coordinates(),
        conch_v1_encoder=_mock_encoder(),
        bcss_masks={
            "mask_tumor": np.ones(N_PATCHES, dtype=np.float32),
            "roi_coverage": np.ones(N_PATCHES, dtype=bool),
        },
        pseudo_gt_cache=_mock_cache(pseudo),
        grounding_source="conch_only",
    )
    assert not used_bcss
    np.testing.assert_array_equal(target, pseudo)


# ─── build_target: bcss_hybrid mode ──────────────────────────────────────────


def test_bcss_used_for_tumor_sentence() -> None:
    """When BCSS has strong tumor signal, it should override pseudo-GT."""
    pseudo = _uniform_pseudo_gt()

    # All patches inside ROI, first 10 are pure tumor
    mask_tumor = np.zeros(N_PATCHES, dtype=np.float32)
    mask_tumor[:10] = 1.0  # above threshold
    roi_coverage = np.ones(N_PATCHES, dtype=bool)

    bcss_masks = {
        "mask_tumor": mask_tumor,
        "mask_stroma": np.zeros(N_PATCHES, dtype=np.float32),
        "mask_lymph": np.zeros(N_PATCHES, dtype=np.float32),
        "mask_necrosis": np.zeros(N_PATCHES, dtype=np.float32),
        "roi_coverage": roi_coverage,
    }

    target, used_bcss = build_target(
        slide_id="TCGA-TEST",
        sentence="Invasive ductal carcinoma grade 3",
        patch_embeddings_v1=_patch_embeddings(),
        coordinates=_coordinates(),
        conch_v1_encoder=_mock_encoder(),
        bcss_masks=bcss_masks,
        pseudo_gt_cache=_mock_cache(pseudo),
        grounding_source="bcss_hybrid",
    )

    assert used_bcss
    assert abs(target.sum() - 1.0) < 1e-5
    # Non-tumor patches should have zero weight
    assert np.all(target[10:] == 0.0)


def test_pseudo_gt_fallback_when_no_bcss() -> None:
    """No BCSS masks → always pseudo-GT."""
    pseudo = _uniform_pseudo_gt()
    target, used_bcss = build_target(
        slide_id="TCGA-TEST",
        sentence="Invasive ductal carcinoma",
        patch_embeddings_v1=_patch_embeddings(),
        coordinates=_coordinates(),
        conch_v1_encoder=_mock_encoder(),
        bcss_masks=None,
        pseudo_gt_cache=_mock_cache(pseudo),
        grounding_source="bcss_hybrid",
    )
    assert not used_bcss
    np.testing.assert_array_equal(target, pseudo)


def test_pseudo_gt_fallback_when_no_concept_match() -> None:
    """Sentence without a concept keyword → pseudo-GT even if BCSS is present."""
    pseudo = _uniform_pseudo_gt()
    bcss_masks = {
        "mask_tumor": np.ones(N_PATCHES, dtype=np.float32),
        "roi_coverage": np.ones(N_PATCHES, dtype=bool),
    }
    target, used_bcss = build_target(
        slide_id="TCGA-TEST",
        sentence="Margins are negative for malignancy",
        patch_embeddings_v1=_patch_embeddings(),
        coordinates=_coordinates(),
        conch_v1_encoder=_mock_encoder(),
        bcss_masks=bcss_masks,
        pseudo_gt_cache=_mock_cache(pseudo),
        grounding_source="bcss_hybrid",
    )
    # "malignancy" is not in CONCEPT_TO_BCSS_KEY, "margins" is not either
    assert not used_bcss


def test_pseudo_gt_fallback_when_roi_coverage_sparse() -> None:
    """If only a tiny fraction of patches are in the BCSS ROI, use pseudo-GT."""
    pseudo = _uniform_pseudo_gt()
    roi_coverage = np.zeros(N_PATCHES, dtype=bool)
    roi_coverage[0] = True  # only 1/50 = 2% coverage < MIN_BCSS_COVERAGE_FRACTION

    bcss_masks = {
        "mask_tumor": np.array([1.0] + [0.0] * (N_PATCHES - 1), dtype=np.float32),
        "roi_coverage": roi_coverage,
    }
    target, used_bcss = build_target(
        slide_id="TCGA-TEST",
        sentence="Invasive carcinoma present",
        patch_embeddings_v1=_patch_embeddings(),
        coordinates=_coordinates(),
        conch_v1_encoder=_mock_encoder(),
        bcss_masks=bcss_masks,
        pseudo_gt_cache=_mock_cache(pseudo),
        grounding_source="bcss_hybrid",
    )
    assert not used_bcss


def test_target_distribution_sums_to_one() -> None:
    """Regardless of routing, target must sum to 1."""
    pseudo = _uniform_pseudo_gt()
    mask_tumor = np.zeros(N_PATCHES, dtype=np.float32)
    mask_tumor[5:15] = 0.8

    bcss_masks = {
        "mask_tumor": mask_tumor,
        "mask_stroma": np.zeros(N_PATCHES, dtype=np.float32),
        "mask_lymph": np.zeros(N_PATCHES, dtype=np.float32),
        "mask_necrosis": np.zeros(N_PATCHES, dtype=np.float32),
        "roi_coverage": np.ones(N_PATCHES, dtype=bool),
    }
    target, _ = build_target(
        slide_id="TCGA-TEST",
        sentence="Tumor cells with lymphocytic stroma",
        patch_embeddings_v1=_patch_embeddings(),
        coordinates=_coordinates(),
        conch_v1_encoder=_mock_encoder(),
        bcss_masks=bcss_masks,
        pseudo_gt_cache=_mock_cache(pseudo),
        grounding_source="bcss_hybrid",
    )
    assert abs(target.sum() - 1.0) < 1e-5
