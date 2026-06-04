"""One-shot orchestrator: build the 50-slide BCSS-covered fullscale manifest
(reusing the existing 10 smoke slides, picking 40 new ones), patch every
file_id against the live GDC API, then emit ROI bounds + a 70/15/15 instruction
split — all in a single deterministic Python pass.

Outputs (under data/metadata/ and data/processed/slideinstruction_fullscale/):
    fullscale_manifest.tsv        50 rows {slide_id, file_id, file_name, mag, has_caption}
    fullscale_bcss_rois.tsv       ROI bounds for the new 40 (existing 10 already downloaded)
    slideinstruction_fullscale/{train,val,test}.json

Usage:
    uv run python scripts/fullscale_prepare.py --limit 50
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import time
import urllib.request
from pathlib import Path

SLIDE_MAG = "https://raw.githubusercontent.com/PathologyDataScience/BCSS/master/meta/slide_magnifications.csv"
ROI_BOUNDS = "https://raw.githubusercontent.com/PathologyDataScience/BCSS/master/meta/roiBounds.csv"
BARCODE = re.compile(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+-\d+[A-Z]-\d+-DX\d+)")


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "patholens"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()


def patient_dx_key(barcode: str) -> str:
    parts = barcode.split("-")
    return f"{parts[0]}-{parts[1]}-{parts[2]}-{parts[5]}"


def fetch_bcss_slides() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in csv.DictReader(io.StringIO(_get(SLIDE_MAG))):
        m = BARCODE.search(row["name"])
        if m:
            out[m.group(1)] = {"file_name": row["name"], "file_id": "", "mag": row.get("magnification", "")}
    return out


def fetch_roi_bounds() -> dict[str, list[dict]]:
    """First column is patient_dx_key like 'TCGA-A1-A0SK-DX1'."""
    out: dict[str, list[dict]] = {}
    reader = csv.reader(io.StringIO(_get(ROI_BOUNDS)))
    next(reader)  # header: ['', 'xmin', 'ymin', 'xmax', 'ymax', 'mask_link']
    for row in reader:
        if not row or not row[0]:
            continue
        out.setdefault(row[0], []).append({
            "xmin": int(row[1]), "ymin": int(row[2]),
            "xmax": int(row[3]), "ymax": int(row[4]),
            "mask_link": row[5],
        })
    return out


def fetch_slidechat_brca_barcodes(directory: Path) -> set[str]:
    barcodes: set[str] = set()
    for split in ("train", "val", "test"):
        fp = directory / f"{split}.json"
        if not fp.exists():
            continue
        for ex in json.load(open(fp)):
            img = ex["image"][0] if isinstance(ex["image"], list) else ex["image"]
            if "/BRCA/" not in img:
                continue
            m = BARCODE.search(img)
            if m:
                barcodes.add(m.group(1))
    return barcodes


def patch_gdc_fileids(manifest: list[dict]) -> list[dict]:
    """For each manifest row, query GDC files API by file_name and update file_id."""
    mapping: dict[str, dict] = {}
    page = 0
    total = None
    while True:
        payload = {
            "filters": {"op": "and", "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
                {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
                {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
                {"op": "in", "content": {"field": "access", "value": ["open"]}},
            ]},
            "fields": "file_id,file_name,file_size",
            "size": 200, "from": page * 200,
        }
        req = urllib.request.Request(
            "https://api.gdc.cancer.gov/files",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        r = json.loads(urllib.request.urlopen(req, timeout=60).read())
        if total is None:
            total = r["data"]["pagination"]["total"]
        hits = r["data"]["hits"]
        if not hits:
            break
        for h in hits:
            m = BARCODE.search(h["file_name"])
            if m:
                mapping[m.group(1)] = {"file_id": h["file_id"], "file_name": h["file_name"]}
        page += 1
        if page * 200 >= total:
            break
        time.sleep(0.2)
    print(f"  GDC barcodes available: {len(mapping)}")

    patched: list[dict] = []
    missing: list[str] = []
    for row in manifest:
        b = row["slide_id"]
        if b in mapping:
            row["file_id"] = mapping[b]["file_id"]
            row["file_name"] = mapping[b]["file_name"]
            patched.append(row)
        else:
            missing.append(b)
    if missing:
        print(f"  [warn] {len(missing)} slides missing from GDC: {missing[:5]}…")
    return patched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--slideinstruction-dir", default="data/processed/slideinstruction")
    ap.add_argument("--brca-instr-dir", default="data/processed/slideinstruction_brca")
    ap.add_argument("--manifest-out", default="data/metadata/fullscale_manifest.tsv")
    ap.add_argument("--roi-out", default="data/metadata/fullscale_bcss_rois.tsv")
    ap.add_argument("--instr-out", default="data/processed/slideinstruction_fullscale")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"=== Fullscale preparation (limit={args.limit}) ===\n")

    print("→ Fetching BCSS slide magnifications…")
    bcss = fetch_bcss_slides()
    print(f"  BCSS slides: {len(bcss)}")

    print("→ Fetching BCSS ROI bounds…")
    rois = fetch_roi_bounds()
    print(f"  ROI entries: {sum(len(v) for v in rois.values())} across {len(rois)} keys")

    print("→ Reading SlideChat BRCA barcodes…")
    sc = fetch_slidechat_brca_barcodes(Path(args.slideinstruction_dir))
    print(f"  SlideChat BRCA slides: {len(sc)}")

    candidates = sorted([
        b for b in bcss
        if b in sc and patient_dx_key(b) in rois
    ])
    print(f"\n  Intersection (BCSS ∩ SlideChat ∩ ROIbounds): {len(candidates)} slides")
    chosen = candidates[: args.limit]
    print(f"  Selected first {len(chosen)} alphabetically (deterministic).\n")

    print("→ Querying GDC for current file_ids…")
    manifest_in = [{
        "slide_id": b, "file_id": "",
        "file_name": bcss[b]["file_name"], "mag": bcss[b]["mag"],
    } for b in chosen]
    manifest = patch_gdc_fileids(manifest_in)
    print(f"  Patched: {len(manifest)}/{len(chosen)}\n")

    out_path = Path(args.manifest_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("slide_id\tfile_id\tfile_name\tmag\thas_caption\n")
        for r in manifest:
            f.write(f"{r['slide_id']}\t{r['file_id']}\t{r['file_name']}\t{r['mag']}\t1\n")
    print(f"✓ Manifest → {out_path}")

    roi_path = Path(args.roi_out)
    with open(roi_path, "w") as f:
        f.write("slide_id\txmin\tymin\txmax\tymax\tmask_link\n")
        for r in manifest:
            for roi in rois[patient_dx_key(r["slide_id"])]:
                f.write(
                    f"{r['slide_id']}\t{roi['xmin']}\t{roi['ymin']}\t"
                    f"{roi['xmax']}\t{roi['ymax']}\t{roi['mask_link']}\n"
                )
    print(f"✓ ROI bounds → {roi_path}")

    # ── Instruction set: read pre-converted BRCA records and split 70/15/15
    print("\n→ Building train/val/test instruction set…")
    pool: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        fp = Path(args.brca_instr_dir) / f"{split}.json"
        if fp.exists():
            for ex in json.load(open(fp)):
                pool[ex["slide_id"]] = ex

    ordered = [pool[r["slide_id"]] for r in manifest if r["slide_id"] in pool]
    print(f"  Matched {len(ordered)}/{len(manifest)} slides to converted captions")

    random.Random(args.seed).shuffle(ordered)
    n = len(ordered)
    n_tr = int(round(n * 0.70))
    n_val = max(1, int(round(n * 0.15)))
    n_test = n - n_tr - n_val
    train, val, test = ordered[:n_tr], ordered[n_tr:n_tr + n_val], ordered[n_tr + n_val:]

    out_dir = Path(args.instr_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(train, open(out_dir / "train.json", "w"), indent=2)
    json.dump(val,   open(out_dir / "val.json",   "w"), indent=2)
    json.dump(test,  open(out_dir / "test.json",  "w"), indent=2)
    print(f"  train={len(train)}  val={len(val)}  test={len(test)} → {out_dir}")
    print(f"\n=== DONE ===")


if __name__ == "__main__":
    main()
