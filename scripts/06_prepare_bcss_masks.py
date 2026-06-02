"""Prepare BCSS patch-level grounding masks for the smoke set.

Two phases:
  download : pull each slide's figshare mask PNG, save under masks/ with the
             filename convention BcssLoader expects
             ({slide_id}_xmin{X}_ymin{Y}_MPP-0.25.png).
  build    : for every slide that has a precomputed embedding HDF5 (coordinates),
             compute per-patch class coverage and write
             data/processed/bcss_patch_masks/{slide_id}.h5.

Run `download` any time; run `build` after precompute_embeddings has produced
the per-slide embedding HDF5 files.

Usage:
    python scripts/06_prepare_bcss_masks.py download
    python scripts/06_prepare_bcss_masks.py build
    python scripts/06_prepare_bcss_masks.py all      # download then build
"""

from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path

ROI_TSV = Path("data/metadata/smoke_bcss_rois.tsv")
RAW_MASK_DIR = Path("data/raw/bcss_masks")
EMB_DIR = Path("data/processed/embeddings/tcga_brca")
OUT_DIR = Path("data/processed/bcss_patch_masks")


def phase_download() -> None:
    RAW_MASK_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROI_TSV), delimiter="\t"))
    print(f"Downloading {len(rows)} BCSS ROI masks...")
    for r in rows:
        sid = r["slide_id"]
        fname = f"{sid}_xmin{r['xmin']}_ymin{r['ymin']}_MPP-0.25.png"
        dest = RAW_MASK_DIR / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  SKIP {fname}")
            continue
        print(f"  -> {fname} ({r['mask_link']})")
        try:
            urllib.request.urlretrieve(r["mask_link"], dest)
        except Exception as e:
            print(f"  FAILED {sid}: {e}")
    pngs = list(RAW_MASK_DIR.glob("*.png"))
    print(f"Have {len(pngs)} mask PNGs in {RAW_MASK_DIR}")


def phase_build() -> None:
    import h5py

    from patholens.data.bcss_loader import BcssLoader

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loader = BcssLoader(RAW_MASK_DIR)
    print(f"BcssLoader indexed {len(loader.slide_ids)} slides")

    built = 0
    for emb in sorted(EMB_DIR.glob("*.h5")):
        slide_id = emb.stem
        if slide_id not in loader.slide_ids:
            continue
        with h5py.File(emb, "r") as f:
            coords = f["coordinates"][:]
        out_path = OUT_DIR / f"{slide_id}.h5"
        loader.save_slide_h5(slide_id=slide_id, coordinates=coords, output_path=out_path)
        with h5py.File(out_path, "r") as f:
            n_roi = int(f["roi_coverage"][:].sum())
        print(f"  {slide_id}: {len(coords)} patches, {n_roi} ROI-covered -> {out_path}")
        built += 1
    print(f"Built {built} BCSS patch-mask files in {OUT_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["download", "build", "all"])
    args = ap.parse_args()
    if args.phase in ("download", "all"):
        phase_download()
    if args.phase in ("build", "all"):
        phase_build()


if __name__ == "__main__":
    main()
