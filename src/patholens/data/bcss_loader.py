"""BCSS (Breast Cancer Semantic Segmentation) annotation loader.

Downloads CC0 annotation masks from the PathologyDataScience/BCSS GitHub
repository and converts per-slide 40× PNG label maps into patch-level class
coverage arrays aligned with our 20× WSI patch grid.

Each BCSS mask is a PNG whose pixel values are class codes 0–21. The ROI
position within the source SVS is encoded in the filename:
    TCGA-A1-7519-01Z-00-DX1_xmin43088_ymin17788_MPP-0.25.png

References:
    Amgad et al. (2019). Structured crowdsourcing enables convolutional
    segmentation of histology images. Bioinformatics 35(18):3461-3467.
    https://github.com/PathologyDataScience/BCSS
"""

from __future__ import annotations

import logging
import re
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ─── BCSS class codes ────────────────────────────────────────────────────────
OUTSIDE_ROI = 0
TUMOR_CLASS = 1
STROMA_CLASS = 2
LYMPH_CLASS = 3  # lymphocytic infiltrate
NECROSIS_CLASS = 4  # necrosis or debris

# Minimum fraction of BCSS pixels in a patch to consider the patch annotated
ROI_COVERAGE_MIN_FRACTION = 0.05

# BCSS native resolution: 40× = 0.25 μm/px (same as TCGA-BRCA level-0)
BCSS_MPP = 0.25

# GitHub sparse-checkout target: only the masks/ subdirectory
BCSS_GITHUB_REPO = "https://github.com/PathologyDataScience/BCSS.git"
BCSS_MASKS_SUBDIR = "masks"

# Filename regex: handles optional _mag-40 and optional -labels suffix
_FILENAME_RE = re.compile(
    r"^(TCGA-[A-Z0-9-]+)_xmin(\d+)_ymin(\d+)_MPP-[\d.]+" r"(?:_mag-\d+)?(?:-labels)?\.png$"
)


# ─── Data structures ──────────────────────────────────────────────────────────


@dataclass
class BcssRoi:
    """One annotated ROI for a slide.

    Attributes:
        slide_id: TCGA slide identifier (e.g. TCGA-A1-7519-01Z-00-DX1)
        xmin: ROI left edge in level-0 (40×) coordinates
        ymin: ROI top edge in level-0 (40×) coordinates
        mask_path: path to the PNG label map
        _mask: lazy-loaded (H_roi, W_roi) uint8 class-code array
    """

    slide_id: str
    xmin: int
    ymin: int
    mask_path: Path
    _mask: NDArray | None = field(default=None, repr=False)

    @property
    def mask(self) -> NDArray:
        """Load and cache the PNG label map on first access."""
        if self._mask is None:
            import cv2

            raw = cv2.imread(str(self.mask_path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise OSError(f"Failed to read BCSS mask: {self.mask_path}")
            # Some exports are RGB with class code in the red channel; some are
            # grayscale uint8 directly. Handle both.
            if raw.ndim == 3:
                self._mask = raw[:, :, 0].astype(np.uint8)
            else:
                self._mask = raw.astype(np.uint8)
        return self._mask

    @property
    def width(self) -> int:
        return self.mask.shape[1]

    @property
    def height(self) -> int:
        return self.mask.shape[0]


# ─── Main loader ─────────────────────────────────────────────────────────────


class BcssLoader:
    """Loads BCSS annotations from a local masks directory.

    Scans `mask_dir` for `*.png` files matching the BCSS filename convention,
    then exposes helpers to build per-patch class coverage arrays.

    Args:
        mask_dir: directory containing BCSS PNG label maps
    """

    def __init__(self, mask_dir: str | Path) -> None:
        self.mask_dir = Path(mask_dir)
        self._rois: dict[str, list[BcssRoi]] = {}
        self._scan()

    def _scan(self) -> None:
        """Index all valid BCSS mask files in mask_dir."""
        count = 0
        for png_path in sorted(self.mask_dir.glob("*.png")):
            m = _FILENAME_RE.match(png_path.name)
            if m is None:
                logger.debug("Skipping non-BCSS file: %s", png_path.name)
                continue
            slide_id, xmin_str, ymin_str = m.group(1), m.group(2), m.group(3)
            roi = BcssRoi(
                slide_id=slide_id,
                xmin=int(xmin_str),
                ymin=int(ymin_str),
                mask_path=png_path,
            )
            self._rois.setdefault(slide_id, []).append(roi)
            count += 1
        logger.info("BcssLoader: indexed %d ROI masks for %d slides", count, len(self._rois))

    @property
    def slide_ids(self) -> set[str]:
        """Set of slide IDs with at least one BCSS annotation."""
        return set(self._rois.keys())

    def get_rois(self, slide_id: str) -> list[BcssRoi]:
        """Return all BCSS ROIs for a slide (empty list if none)."""
        return self._rois.get(slide_id, [])

    def build_patch_masks(
        self,
        slide_id: str,
        coordinates: NDArray,
        patch_size: int = 448,
        level_downsample: float = 2.0,
    ) -> dict[str, NDArray]:
        """Compute per-patch class coverage fractions for one slide.

        For each patch, we find all BCSS ROIs that overlap and aggregate
        class fractions weighted by overlap area.

        Args:
            slide_id: TCGA slide identifier
            coordinates: (N_patches, 2) int32 — (x, y) patch origins in level-0
            patch_size: patch edge length in pixels at the extraction magnification
            level_downsample: level-0 pixels per extraction-level pixel
                (2.0 for 20× extraction from 40× native TCGA slides)

        Returns:
            dict with keys:
                mask_tumor:   (N_patches,) float32 — fraction of patch labeled tumor
                mask_stroma:  (N_patches,) float32
                mask_lymph:   (N_patches,) float32
                mask_necrosis:(N_patches,) float32
                roi_coverage: (N_patches,) bool — True if any BCSS ROI covers ≥ 5% of patch
        """
        n_patches = len(coordinates)
        mask_tumor = np.zeros(n_patches, dtype=np.float32)
        mask_stroma = np.zeros(n_patches, dtype=np.float32)
        mask_lymph = np.zeros(n_patches, dtype=np.float32)
        mask_necrosis = np.zeros(n_patches, dtype=np.float32)
        roi_coverage = np.zeros(n_patches, dtype=bool)

        rois = self.get_rois(slide_id)
        if not rois:
            return {
                "mask_tumor": mask_tumor,
                "mask_stroma": mask_stroma,
                "mask_lymph": mask_lymph,
                "mask_necrosis": mask_necrosis,
                "roi_coverage": roi_coverage,
            }

        # Patch footprint in level-0 / BCSS-pixel space
        patch_size_l0 = int(round(patch_size * level_downsample))

        # Accumulators for weighted averaging across multiple ROIs
        weight_sum = np.zeros(n_patches, dtype=np.float32)
        accum = {
            TUMOR_CLASS: np.zeros(n_patches, dtype=np.float32),
            STROMA_CLASS: np.zeros(n_patches, dtype=np.float32),
            LYMPH_CLASS: np.zeros(n_patches, dtype=np.float32),
            NECROSIS_CLASS: np.zeros(n_patches, dtype=np.float32),
        }

        for roi in rois:
            mask = roi.mask  # (H_roi, W_roi) uint8

            for idx, (x_l0, y_l0) in enumerate(coordinates):
                # Patch bounds in BCSS pixel space (same as level-0 for 40× native)
                col0 = int(x_l0) - roi.xmin
                row0 = int(y_l0) - roi.ymin
                col1 = col0 + patch_size_l0
                row1 = row0 + patch_size_l0

                # Clip to ROI bounds
                col0c = max(col0, 0)
                row0c = max(row0, 0)
                col1c = min(col1, roi.width)
                row1c = min(row1, roi.height)

                if col0c >= col1c or row0c >= row1c:
                    continue  # no overlap

                patch_roi = mask[row0c:row1c, col0c:col1c]
                total_pixels = patch_roi.size
                if total_pixels == 0:
                    continue

                annotated_pixels = int((patch_roi != OUTSIDE_ROI).sum())
                annotated_frac = annotated_pixels / total_pixels

                if annotated_frac < ROI_COVERAGE_MIN_FRACTION:
                    continue  # mostly background — skip

                roi_coverage[idx] = True
                w = float(annotated_pixels)
                weight_sum[idx] += w

                for cls_code, arr in accum.items():
                    arr[idx] += float((patch_roi == cls_code).sum())

        # Normalise accumulators to fractions
        nonzero = weight_sum > 0
        for _cls_code, arr in accum.items():
            arr[nonzero] /= weight_sum[nonzero]

        mask_tumor = accum[TUMOR_CLASS]
        mask_stroma = accum[STROMA_CLASS]
        mask_lymph = accum[LYMPH_CLASS]
        mask_necrosis = accum[NECROSIS_CLASS]

        return {
            "mask_tumor": mask_tumor,
            "mask_stroma": mask_stroma,
            "mask_lymph": mask_lymph,
            "mask_necrosis": mask_necrosis,
            "roi_coverage": roi_coverage,
        }

    def save_slide_h5(
        self,
        slide_id: str,
        coordinates: NDArray,
        output_path: str | Path,
        patch_size: int = 448,
        level_downsample: float = 2.0,
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> Path:
        """Compute patch masks and write to HDF5.

        Args:
            slide_id: TCGA slide identifier
            coordinates: (N_patches, 2) int32 — patch origins in level-0
            output_path: path for output HDF5 file
            patch_size: extraction patch size in pixels
            level_downsample: level-0 pixels per extraction-level pixel
            compression: HDF5 compression filter
            compression_opts: HDF5 compression level

        Returns:
            output_path as Path
        """
        masks = self.build_patch_masks(slide_id, coordinates, patch_size, level_downsample)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        kwargs = {"compression": compression, "compression_opts": compression_opts}
        with h5py.File(output_path, "w") as f:
            for key, arr in masks.items():
                f.create_dataset(key, data=arr, **kwargs)
            f.attrs["slide_id"] = slide_id
            f.attrs["n_patches"] = len(coordinates)
            f.attrs["n_roi_patches"] = int(masks["roi_coverage"].sum())
            f.attrs["n_rois"] = len(self.get_rois(slide_id))

        return output_path


# ─── Download helper ─────────────────────────────────────────────────────────


def download_bcss_masks(output_dir: str | Path, overwrite: bool = False) -> Path:
    """Download BCSS mask PNGs from the official GitHub repository.

    Uses `git sparse-checkout` to pull only the `masks/` subdirectory
    (~300 MB). Requires git ≥ 2.25 on PATH.

    Args:
        output_dir: local directory to write mask PNGs into
        overwrite: if True, re-download even if output_dir already exists

    Returns:
        Path to the directory containing the downloaded *.png files
    """
    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"

    if masks_dir.exists() and not overwrite:
        existing = list(masks_dir.glob("*.png"))
        if existing:
            logger.info(
                "BCSS masks already present (%d files) — skipping download. "
                "Pass overwrite=True to re-download.",
                len(existing),
            )
            return masks_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    clone_dir = output_dir / "_bcss_sparse_clone"
    clone_dir.mkdir(exist_ok=True)

    try:
        # Init a bare repo and enable sparse checkout
        subprocess.run(["git", "init"], cwd=clone_dir, check=True, capture_output=True)
        subprocess.run(
            ["git", "remote", "add", "origin", BCSS_GITHUB_REPO],
            cwd=clone_dir,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "init", "--cone"],
            cwd=clone_dir,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", BCSS_MASKS_SUBDIR],
            cwd=clone_dir,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "pull", "--depth=1", "origin", "master"],
            cwd=clone_dir,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"git sparse-checkout failed: {exc}\n"
            "Manual alternative:\n"
            f"  git clone --filter=blob:none --sparse {BCSS_GITHUB_REPO} {clone_dir}\n"
            f"  cd {clone_dir} && git sparse-checkout set masks"
        ) from exc

    # Move PNGs to output_dir/masks/
    src = clone_dir / BCSS_MASKS_SUBDIR
    if not src.exists():
        raise FileNotFoundError(f"Expected masks/ directory not found after clone: {src}")
    masks_dir.mkdir(exist_ok=True)
    for png in src.glob("*.png"):
        png.rename(masks_dir / png.name)

    import shutil

    shutil.rmtree(clone_dir, ignore_errors=True)

    downloaded = list(masks_dir.glob("*.png"))
    logger.info("Downloaded %d BCSS mask files to %s", len(downloaded), masks_dir)
    if len(downloaded) == 0:
        warnings.warn(
            "No PNG files found after BCSS download — check git clone output.",
            stacklevel=2,
        )
    return masks_dir
