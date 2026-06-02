"""Select a smoke-test set of TCGA-BRCA DX slides for the end-to-end pipeline run.

The set is the intersection of:
  - BCSS annotated slides (pixel masks via figshare → real grounding GT)
  - SlideChat BRCA slides (captions/VQA → caption training target)

BCSS metadata gives us everything we need without hitting the GDC API:
  - meta/slide_magnifications.csv : full GDC filename (UUID embedded) per slide
  - meta/roiBounds.csv            : ROI xmin/ymin + figshare mask download link

Outputs:
  data/metadata/smoke_manifest.tsv   slide_id, file_id, file_name, mag
  data/metadata/smoke_bcss_rois.tsv  slide_id, xmin, ymin, xmax, ymax, mask_link

slide_id is the bare barcode (e.g. TCGA-A1-A0SK-01Z-00-DX1). The download step
saves each SVS as {slide_id}.svs so embeddings, BCSS masks, and captions all key
on the same barcode.

Usage:
    python scripts/select_smoke_set.py --limit 10
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import urllib.request
from pathlib import Path

SLIDE_MAG_CSV = "https://raw.githubusercontent.com/PathologyDataScience/BCSS/master/meta/slide_magnifications.csv"
ROI_BOUNDS_CSV = "https://raw.githubusercontent.com/PathologyDataScience/BCSS/master/meta/roiBounds.csv"
BARCODE_RE = re.compile(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+-\d+[A-Z]-\d+-DX\d+)")


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "patholens"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()


def fetch_bcss_slides() -> dict[str, dict]:
    """Return {barcode: {file_name, file_id, mag}} from slide_magnifications.csv."""
    text = _get(SLIDE_MAG_CSV)
    out: dict[str, dict] = {}
    for row in csv.DictReader(io.StringIO(text)):
        fname = row["name"]
        m = BARCODE_RE.search(fname)
        if not m:
            continue
        barcode = m.group(1)
        # filename: TCGA-A1-A0SK-01Z-00-DX1.<UUID>.svs → file_id = <UUID> lowercased
        rest = fname[len(barcode) + 1 :]  # strip "barcode."
        file_id = rest.rsplit(".svs", 1)[0].lower()
        out[barcode] = {"file_name": fname, "file_id": file_id, "mag": row.get("magnification", "")}
    return out


def fetch_roi_bounds() -> dict[str, list[dict]]:
    """Return {patient_dx_key: [{xmin,ymin,xmax,ymax,mask_link}, ...]}.

    roiBounds.csv keys look like 'TCGA-A1-A0SK-DX1'.
    """
    text = _get(ROI_BOUNDS_CSV)
    out: dict[str, list[dict]] = {}
    reader = csv.reader(io.StringIO(text))
    header = next(reader)  # ['', 'xmin', 'ymin', 'xmax', 'ymax', 'mask_link']
    for row in reader:
        if not row or not row[0]:
            continue
        key = row[0]
        out.setdefault(key, []).append(
            {
                "xmin": int(row[1]),
                "ymin": int(row[2]),
                "xmax": int(row[3]),
                "ymax": int(row[4]),
                "mask_link": row[5],
            }
        )
    return out


def patient_dx_key(barcode: str) -> str:
    """TCGA-A1-A0SK-01Z-00-DX1 → TCGA-A1-A0SK-DX1 (roiBounds key format)."""
    p = barcode.split("-")
    return f"{p[0]}-{p[1]}-{p[2]}-{p[5]}"


def fetch_slidechat_brca_barcodes(slideinstruction_dir: Path) -> set[str]:
    barcodes: set[str] = set()
    for split in ("train", "val", "test"):
        fp = slideinstruction_dir / f"{split}.json"
        if not fp.exists():
            continue
        for ex in json.load(open(fp)):
            img = ex["image"][0] if isinstance(ex["image"], list) else ex["image"]
            if "/BRCA/" not in img:
                continue
            m = BARCODE_RE.search(img)
            if m:
                barcodes.add(m.group(1))
    return barcodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--slideinstruction-dir", default="data/processed/slideinstruction")
    ap.add_argument("--out", default="data/metadata/smoke_manifest.tsv")
    ap.add_argument("--roi-out", default="data/metadata/smoke_bcss_rois.tsv")
    args = ap.parse_args()

    print("→ Fetching BCSS slide list (slide_magnifications.csv)...")
    bcss = fetch_bcss_slides()
    print(f"  BCSS slides: {len(bcss)}")

    print("→ Fetching BCSS ROI bounds (roiBounds.csv)...")
    rois = fetch_roi_bounds()
    print(f"  ROI entries: {sum(len(v) for v in rois.values())} across {len(rois)} keys")

    print("→ Extracting SlideChat BRCA barcodes...")
    sc = fetch_slidechat_brca_barcodes(Path(args.slideinstruction_dir))
    print(f"  SlideChat BRCA slides: {len(sc)}")

    # Intersection: BCSS slides that also have SlideChat captions AND ROI bounds
    candidates = [
        b for b in bcss if b in sc and patient_dx_key(b) in rois
    ]
    candidates.sort()
    print(f"\n  Intersection (BCSS ∩ SlideChat ∩ ROIbounds): {len(candidates)} slides")

    if not candidates:
        print("  [warn] empty intersection — falling back to BCSS ∩ ROIbounds")
        candidates = sorted(b for b in bcss if patient_dx_key(b) in rois)

    chosen = candidates[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("slide_id\tfile_id\tfile_name\tmag\thas_caption\n")
        for b in chosen:
            g = bcss[b]
            f.write(f"{b}\t{g['file_id']}\t{g['file_name']}\t{g['mag']}\t{int(b in sc)}\n")

    roi_path = Path(args.roi_out)
    with open(roi_path, "w") as f:
        f.write("slide_id\txmin\tymin\txmax\tymax\tmask_link\n")
        for b in chosen:
            for roi in rois[patient_dx_key(b)]:
                f.write(
                    f"{b}\t{roi['xmin']}\t{roi['ymin']}\t{roi['xmax']}\t"
                    f"{roi['ymax']}\t{roi['mask_link']}\n"
                )

    print(f"\n✓ Wrote {len(chosen)} slides → {out_path}")
    print(f"✓ Wrote ROI bounds → {roi_path}")
    for b in chosen:
        print(f"    {b}  caption={int(b in sc)}  rois={len(rois[patient_dx_key(b)])}")


if __name__ == "__main__":
    main()
