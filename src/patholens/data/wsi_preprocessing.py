"""WSI preprocessing: tissue segmentation + patch extraction.

Pipeline:
    1. Load WSI with openslide
    2. Tissue segmentation (Otsu @ 1.25x with GrandQC fallback)
    3. Patch extraction (448x448 @20x from tissue regions)
    4. Return patches + coordinates

To be implemented by Claude Code in Hafta 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def otsu_tissue_mask(thumbnail: NDArray, sat_threshold: int = 8) -> NDArray:
    """Compute tissue mask from low-resolution WSI thumbnail using Otsu on HSV saturation.

    Args:
        thumbnail: (H, W, 3) RGB array @1.25x magnification
        sat_threshold: minimum saturation value to be considered tissue

    Returns:
        (H, W) binary mask (1 = tissue)
    """
    raise NotImplementedError("Implement in Hafta 1")


def extract_patches(
    wsi_path: str | Path,
    tissue_mask: NDArray,
    patch_size: int = 448,
    magnification: int = 20,
    tissue_ratio_threshold: float = 0.5,
) -> tuple[NDArray, NDArray]:
    """Extract patches from WSI at given magnification, filtered by tissue mask.

    Args:
        wsi_path: path to .svs/.tif file
        tissue_mask: low-res tissue mask
        patch_size: patch dimension in pixels (at target magnification)
        magnification: target magnification (e.g., 20)
        tissue_ratio_threshold: minimum tissue fraction to keep patch

    Returns:
        patches: (N, patch_size, patch_size, 3) uint8 RGB array
        coordinates: (N, 2) int32 array of (x, y) origin in WSI coords
    """
    raise NotImplementedError("Implement in Hafta 1")
