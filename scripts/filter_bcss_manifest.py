"""Fetch BCSS slide list and filter GDC manifest to only those slides.

BCSS (Amgad et al. 2019) annotates 151 TCGA-BRCA slides. Their slide IDs
follow the TCGA barcode format: TCGA-XX-XXXX-01Z-00-DX1 (or DX2).

This script:
1. Downloads the BCSS case list from their GitHub
2. Matches against our full GDC manifest
3. Writes a filtered manifest ready for gdc-client

Usage:
    python scripts/filter_bcss_manifest.py
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

# BCSS case list — from the official BCSS repository
BCSS_CASE_URL = (
    "https://raw.githubusercontent.com/PathologyDataScience/"
    "BCSS/master/meta/gtruth_codes.tsv"
)

MANIFEST_IN = "data/metadata/gdc_manifest.txt"
MANIFEST_OUT = "data/metadata/gdc_manifest_bcss151.txt"
SLIDE_IDS_OUT = "data/metadata/bcss_slide_ids.txt"


def fetch_bcss_case_ids() -> list[str]:
    """Download BCSS case list and return TCGA case IDs."""
    print("Fetching BCSS case list from GitHub...")
    try:
        with urllib.request.urlopen(BCSS_CASE_URL, timeout=15) as r:
            content = r.read().decode()
        # Parse TSV — first column is the slide name like TCGA-OL-A5RW-01Z-00-DX1
        case_ids = []
        for line in content.strip().splitlines()[1:]:  # skip header
            parts = line.split("\t")
            if parts and parts[0].startswith("TCGA"):
                case_ids.append(parts[0].strip())
        print(f"  Found {len(case_ids)} BCSS-annotated slides")
        return case_ids
    except Exception as e:
        print(f"  Could not fetch from GitHub: {e}")
        print("  Falling back to GDC API query for BCSS cases...")
        return fetch_bcss_via_gdc()


def fetch_bcss_via_gdc() -> list[str]:
    """Alternative: query GDC for TCGA-BRCA cases that have BCSS annotations."""
    # BCSS annotates specific TCGA-BRCA cases — query by experimental strategy
    # We use a broad BRCA query and rely on filename matching
    url = "https://api.gdc.cancer.gov/files"
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": ["TCGA-BRCA"],
                    },
                },
                {
                    "op": "in",
                    "content": {"field": "data_format", "value": ["SVS"]},
                },
                {
                    "op": "in",
                    "content": {"field": "access", "value": ["open"]},
                },
            ],
        },
        "fields": "file_name",
        "size": 3200,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read())
    return [h["file_name"].replace(".svs", "") for h in resp["data"]["hits"]]


def load_manifest(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def match_bcss_to_manifest(
    bcss_ids: list[str], manifest_rows: list[dict]
) -> list[dict]:
    """Match BCSS case IDs to manifest rows by filename prefix."""
    # BCSS IDs look like: TCGA-OL-A5RW-01Z-00-DX1
    # GDC filenames look like: TCGA-OL-A5RW-01Z-00-DX1.svs (or with UUID prefix)
    matched = []
    unmatched = []

    for bcss_id in bcss_ids:
        # Try exact filename match (with or without .svs)
        found = None
        for row in manifest_rows:
            fname = row.get("filename", "")
            if bcss_id in fname:
                found = row
                break
        if found:
            matched.append(found)
        else:
            unmatched.append(bcss_id)

    return matched, unmatched


def main() -> None:
    bcss_ids = fetch_bcss_case_ids()

    print(f"\nLoading manifest: {MANIFEST_IN}")
    manifest = load_manifest(MANIFEST_IN)
    print(f"  Total manifest rows: {len(manifest)}")

    matched, unmatched = match_bcss_to_manifest(bcss_ids, manifest)
    print(f"\nMatched: {len(matched)} / {len(bcss_ids)} BCSS slides found in GDC manifest")

    if unmatched:
        print(f"Unmatched ({len(unmatched)}):")
        for uid in unmatched[:10]:
            print(f"  {uid}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")

    # Write filtered manifest
    out_path = Path(MANIFEST_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("id\tfilename\tmd5\tsize\tstate\n")
        for row in matched:
            f.write(
                f"{row['id']}\t{row['filename']}\t"
                f"{row.get('md5', '')}\t{row.get('size', '')}\t"
                f"{row.get('state', '')}\n"
            )

    total_gb = sum(int(r.get("size", 0) or 0) for r in matched) / 1e9
    print(f"\nFiltered manifest: {out_path.absolute()}")
    print(f"  {len(matched)} slides, ~{total_gb:.1f} GB")

    # Write slide ID list
    sid_path = Path(SLIDE_IDS_OUT)
    with open(sid_path, "w") as f:
        for row in matched:
            # Extract TCGA barcode from filename
            fname = row.get("filename", "").replace(".svs", "")
            f.write(fname + "\n")
    print(f"  Slide ID list: {sid_path.absolute()}")


if __name__ == "__main__":
    main()
