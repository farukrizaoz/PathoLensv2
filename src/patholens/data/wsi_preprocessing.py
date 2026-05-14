"""WSI preprocessing: tissue segmentation + patch extraction."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

# openslide is imported lazily inside functions that open WSI files so that
# this module can be imported on machines where the native library is absent
# (e.g., local macOS dev without libopenslide installed).

THUMBNAIL_MAG: float = 1.25
TARGET_MAG: int = 20


def _native_magnification(wsi: object) -> float:
    """Return level-0 scan magnification; defaults to 40x for TCGA if missing."""
    import openslide  # noqa: PLC0415

    mag_str = wsi.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if mag_str is not None:
        return float(mag_str)
    return 40.0


def _find_level_for_magnification(wsi: object, target_mag: float) -> int:
    """Return the openslide level closest to target_mag without going below it."""
    native_mag = _native_magnification(wsi)
    target_downsample = native_mag / target_mag
    return wsi.get_best_level_for_downsample(target_downsample)


def otsu_tissue_mask(thumbnail: NDArray, sat_threshold: int = 8) -> NDArray:
    """Compute tissue mask from low-resolution WSI thumbnail using Otsu on HSV saturation.

    Args:
        thumbnail: (H, W, 3) RGB uint8 array at thumbnail magnification
        sat_threshold: floor for Otsu threshold; actual = max(otsu, sat_threshold)

    Returns:
        (H, W) uint8 binary mask (1 = tissue, 0 = background)
    """
    hsv = cv2.cvtColor(thumbnail.astype(np.uint8), cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    otsu_val, _ = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(int(otsu_val), sat_threshold)
    mask = (saturation > threshold).astype(np.uint8)

    # Fill holes then remove pepper noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _morphological_tissue_mask(thumbnail: NDArray) -> NDArray:
    """Fallback tissue mask via grayscale Otsu on inverted image.

    Used when the slide is lightly stained and HSV saturation-based Otsu
    yields coverage below min_tissue_area_pct.
    """
    gray = cv2.cvtColor(thumbnail.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    _, mask = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return (mask > 0).astype(np.uint8)


def get_tissue_mask(
    wsi: object,
    sat_threshold: int = 8,
    min_tissue_area_pct: float = 0.05,
    thumbnail_mag: float = THUMBNAIL_MAG,
) -> tuple[NDArray, float]:
    """Generate tissue mask for a WSI at thumbnail resolution.

    Tries Otsu on HSV saturation; falls back to morphological thresholding when
    tissue coverage is below min_tissue_area_pct.

    Args:
        wsi: opened OpenSlide handle
        sat_threshold: minimum saturation floor for Otsu
        min_tissue_area_pct: if coverage < this, switch to fallback
        thumbnail_mag: magnification to extract thumbnail at (default 1.25x)

    Returns:
        mask: (H, W) uint8 binary mask at thumbnail resolution
        downsample: level-0 pixels per thumbnail pixel (for coordinate mapping)
    """
    native_w, native_h = wsi.dimensions
    native_mag = _native_magnification(wsi)
    thumb_w = max(1, int(native_w * thumbnail_mag / native_mag))
    thumb_h = max(1, int(native_h * thumbnail_mag / native_mag))

    thumbnail = np.array(wsi.get_thumbnail((thumb_w, thumb_h)).convert("RGB"))

    mask = otsu_tissue_mask(thumbnail, sat_threshold)
    tissue_fraction = mask.sum() / mask.size

    if tissue_fraction < min_tissue_area_pct:
        mask = _morphological_tissue_mask(thumbnail)

    # Actual downsample may differ slightly from target due to openslide rounding
    actual_thumb_w = thumbnail.shape[1]
    downsample = native_w / actual_thumb_w
    return mask, downsample


def extract_patches(
    wsi_path: str | Path,
    tissue_mask: NDArray,
    mask_downsample: float,
    patch_size: int = 448,
    magnification: int = TARGET_MAG,
    tissue_ratio_threshold: float = 0.5,
) -> tuple[NDArray, NDArray]:
    """Extract non-overlapping patches from WSI at target magnification, filtered by tissue.

    Args:
        wsi_path: path to .svs or .tif file
        tissue_mask: (H, W) uint8 binary mask at thumbnail resolution
        mask_downsample: level-0 pixels per thumbnail pixel (from get_tissue_mask)
        patch_size: patch edge length in pixels at the target magnification
        magnification: desired magnification (e.g., 20)
        tissue_ratio_threshold: minimum tissue fraction in patch to keep it

    Returns:
        patches: (N, patch_size, patch_size, 3) uint8 RGB
        coordinates: (N, 2) int32 — (x, y) patch origin in level-0 coordinates
    """
    import openslide  # noqa: PLC0415

    wsi = openslide.OpenSlide(str(wsi_path))
    level = _find_level_for_magnification(wsi, magnification)
    level_downsample = wsi.level_downsamples[level]
    level_w, level_h = wsi.level_dimensions[level]

    patches_list: list[NDArray] = []
    coords_list: list[tuple[int, int]] = []

    for row_start in range(0, level_h - patch_size + 1, patch_size):
        for col_start in range(0, level_w - patch_size + 1, patch_size):
            # Map patch origin: level coords → level-0 coords → thumbnail coords
            x_l0 = int(col_start * level_downsample)
            y_l0 = int(row_start * level_downsample)
            x_thumb = int(x_l0 / mask_downsample)
            y_thumb = int(y_l0 / mask_downsample)

            patch_thumb = max(1, int(patch_size * level_downsample / mask_downsample))
            x_end = min(x_thumb + patch_thumb, tissue_mask.shape[1])
            y_end = min(y_thumb + patch_thumb, tissue_mask.shape[0])

            if x_thumb >= tissue_mask.shape[1] or y_thumb >= tissue_mask.shape[0]:
                continue

            roi = tissue_mask[y_thumb:y_end, x_thumb:x_end]
            if roi.size == 0 or roi.sum() / roi.size < tissue_ratio_threshold:
                continue

            patch_pil = wsi.read_region(
                location=(x_l0, y_l0),
                level=level,
                size=(patch_size, patch_size),
            )
            patches_list.append(np.array(patch_pil.convert("RGB")))
            coords_list.append((x_l0, y_l0))

    wsi.close()

    if not patches_list:
        empty_patches = np.empty((0, patch_size, patch_size, 3), dtype=np.uint8)
        empty_coords = np.empty((0, 2), dtype=np.int32)
        return empty_patches, empty_coords

    return np.stack(patches_list, axis=0), np.array(coords_list, dtype=np.int32)
