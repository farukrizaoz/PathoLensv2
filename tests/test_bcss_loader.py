"""Tests for BcssLoader: mask parsing, coordinate alignment, patch coverage."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from patholens.data.bcss_loader import (
    _FILENAME_RE,
    LYMPH_CLASS,
    NECROSIS_CLASS,
    STROMA_CLASS,
    TUMOR_CLASS,
    BcssLoader,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _write_mock_mask(
    path: Path,
    width: int,
    height: int,
    fill_class: int = TUMOR_CLASS,
) -> None:
    """Write a solid-colour 8-bit PNG label map."""
    import cv2

    mask = np.full((height, width), fill_class, dtype=np.uint8)
    cv2.imwrite(str(path), mask)


# ─── Filename regex ──────────────────────────────────────────────────────────


def test_filename_regex_standard() -> None:
    m = _FILENAME_RE.match("TCGA-A1-7519-01Z-00-DX1_xmin43088_ymin17788_MPP-0.25.png")
    assert m is not None
    assert m.group(1) == "TCGA-A1-7519-01Z-00-DX1"
    assert m.group(2) == "43088"
    assert m.group(3) == "17788"


def test_filename_regex_with_mag_suffix() -> None:
    m = _FILENAME_RE.match("TCGA-B7-0001-01Z-00-DX2_xmin0_ymin0_MPP-0.25_mag-40.png")
    assert m is not None
    assert m.group(1) == "TCGA-B7-0001-01Z-00-DX2"


def test_filename_regex_labels_suffix() -> None:
    m = _FILENAME_RE.match("TCGA-C5-1234-01Z-00-DX1_xmin100_ymin200_MPP-0.25-labels.png")
    assert m is not None


def test_filename_regex_rejects_non_bcss() -> None:
    assert _FILENAME_RE.match("random_file.png") is None
    assert _FILENAME_RE.match("TCGA-A1-7519-01Z-00-DX1.png") is None


# ─── BcssLoader with empty directory ─────────────────────────────────────────


def test_loader_empty_dir_returns_zero_masks() -> None:
    with tempfile.TemporaryDirectory() as d:
        loader = BcssLoader(d)
        assert len(loader.slide_ids) == 0

        coords = np.array([[0, 0], [5000, 5000]], dtype=np.int32)
        masks = loader.build_patch_masks("TCGA-FAKE", coords)

        assert masks["mask_tumor"].shape == (2,)
        assert np.all(masks["mask_tumor"] == 0)
        assert np.all(~masks["roi_coverage"])


# ─── BcssLoader with mock masks ──────────────────────────────────────────────


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("cv2"),
    reason="opencv-python not installed",
)
class TestBcssLoaderWithMasks:
    """Tests that require cv2 to write mock PNGs."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self.mask_dir = Path(self._tmpdir)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_loader(
        self, slide_id: str, xmin: int, ymin: int, w: int, h: int, cls: int
    ) -> BcssLoader:
        fname = f"{slide_id}_xmin{xmin}_ymin{ymin}_MPP-0.25.png"
        _write_mock_mask(self.mask_dir / fname, width=w, height=h, fill_class=cls)
        return BcssLoader(self.mask_dir)

    def test_slide_ids_detected(self) -> None:
        slide_id = "TCGA-A1-0001-01Z-00-DX1"
        loader = self._make_loader(slide_id, xmin=0, ymin=0, w=2000, h=2000, cls=TUMOR_CLASS)
        assert slide_id in loader.slide_ids

    def test_tumor_mask_full_roi(self) -> None:
        """Patch fully inside all-tumor ROI → mask_tumor ≈ 1.0."""
        slide_id = "TCGA-A1-0001-01Z-00-DX1"
        # ROI at (0, 0), 2000×2000 pixels — entirely tumor
        loader = self._make_loader(slide_id, xmin=0, ymin=0, w=2000, h=2000, cls=TUMOR_CLASS)

        # Patch at (0, 0) in level-0; patch_size=448, level_downsample=2 → 896×896 in BCSS
        coords = np.array([[0, 0]], dtype=np.int32)
        masks = loader.build_patch_masks(slide_id, coords, patch_size=448, level_downsample=2.0)

        assert masks["roi_coverage"][0] is np.bool_(True)
        assert masks["mask_tumor"][0] == pytest.approx(1.0, abs=0.01)
        assert masks["mask_stroma"][0] == pytest.approx(0.0, abs=0.01)

    def test_stroma_mask_full_roi(self) -> None:
        slide_id = "TCGA-B2-0002-01Z-00-DX1"
        loader = self._make_loader(slide_id, xmin=0, ymin=0, w=2000, h=2000, cls=STROMA_CLASS)
        coords = np.array([[0, 0]], dtype=np.int32)
        masks = loader.build_patch_masks(slide_id, coords, patch_size=448, level_downsample=2.0)
        assert masks["mask_stroma"][0] == pytest.approx(1.0, abs=0.01)

    def test_patch_outside_roi_has_no_coverage(self) -> None:
        """Patch far outside the annotated ROI → roi_coverage False, all masks zero."""
        slide_id = "TCGA-A1-0001-01Z-00-DX1"
        # ROI at (0, 0), 500×500 pixels
        loader = self._make_loader(slide_id, xmin=0, ymin=0, w=500, h=500, cls=TUMOR_CLASS)

        # Patch starting at (50000, 50000) — far outside ROI
        coords = np.array([[50000, 50000]], dtype=np.int32)
        masks = loader.build_patch_masks(slide_id, coords, patch_size=448, level_downsample=2.0)

        assert not masks["roi_coverage"][0]
        assert masks["mask_tumor"][0] == 0.0

    def test_roi_coverage_uses_xmin_ymin_offset(self) -> None:
        """Patch inside an offset ROI should be detected correctly."""
        slide_id = "TCGA-C3-0003-01Z-00-DX1"
        xmin, ymin = 10000, 20000
        loader = self._make_loader(slide_id, xmin=xmin, ymin=ymin, w=3000, h=3000, cls=LYMPH_CLASS)

        # Patch exactly at ROI origin
        coords = np.array([[xmin, ymin]], dtype=np.int32)
        masks = loader.build_patch_masks(slide_id, coords, patch_size=448, level_downsample=2.0)

        assert masks["roi_coverage"][0]
        assert masks["mask_lymph"][0] == pytest.approx(1.0, abs=0.01)

    def test_save_slide_h5(self) -> None:
        """Saved HDF5 should contain all expected datasets."""
        import h5py

        slide_id = "TCGA-D4-0004-01Z-00-DX1"
        loader = self._make_loader(slide_id, xmin=0, ymin=0, w=2000, h=2000, cls=NECROSIS_CLASS)

        coords = np.array([[0, 0], [1000, 1000]], dtype=np.int32)
        out_path = self.mask_dir / "test_output.h5"
        loader.save_slide_h5(slide_id, coords, out_path)

        with h5py.File(out_path, "r") as f:
            for key in ["mask_tumor", "mask_stroma", "mask_lymph", "mask_necrosis", "roi_coverage"]:
                assert key in f, f"Missing dataset: {key}"
                assert f[key].shape == (2,)
            assert f.attrs["slide_id"] == slide_id
            assert f.attrs["n_patches"] == 2
