"""Generate GDC manifest for TCGA-BRCA diagnostic slides.

No GDC token required — SVS slides are open access.

Usage:
    python scripts/generate_gdc_manifest.py
    python scripts/generate_gdc_manifest.py --limit 10  # test with 10 slides
    python scripts/generate_gdc_manifest.py --out data/metadata/gdc_manifest.txt
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
PAGE_SIZE = 100


def query_gdc(page: int = 0, size: int = PAGE_SIZE) -> dict:
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
                    "content": {
                        "field": "data_type",
                        "value": ["Slide Image"],
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "data_format",
                        "value": ["SVS"],
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "access",
                        "value": ["open"],
                    },
                },
            ],
        },
        "fields": "file_id,file_name,file_size,md5sum,state",
        "size": size,
        "from": page * size,
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        GDC_FILES_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main(out: str = "data/metadata/gdc_manifest.txt", limit: int | None = None) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Querying GDC API for TCGA-BRCA SVS slides...")
    first = query_gdc(page=0, size=1)
    total = first["data"]["pagination"]["total"]
    print(f"Total available: {total} files")

    if limit:
        total = min(total, limit)
        print(f"Limiting to {total} files")

    all_hits: list[dict] = []
    page = 0
    while len(all_hits) < total:
        batch_size = min(PAGE_SIZE, total - len(all_hits))
        resp = query_gdc(page=page, size=batch_size)
        hits = resp["data"]["hits"]
        if not hits:
            break
        all_hits.extend(hits)
        print(f"  Fetched {len(all_hits)}/{total}", end="\r")
        page += 1
        time.sleep(0.3)

    print(f"\nWriting manifest: {out_path}")
    with open(out_path, "w") as f:
        f.write("id\tfilename\tmd5\tsize\tstate\n")
        for h in all_hits:
            f.write(
                f"{h['file_id']}\t{h['file_name']}\t"
                f"{h.get('md5sum', '')}\t{h.get('file_size', '')}\t"
                f"{h.get('state', '')}\n"
            )

    total_gb = sum(h.get("file_size", 0) for h in all_hits) / 1e9
    print(f"Done. {len(all_hits)} files, {total_gb:.1f} GB total.")
    print(f"Manifest saved to: {out_path.absolute()}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/metadata/gdc_manifest.txt")
    p.add_argument("--limit", type=int, default=None, help="Limit number of files (for testing)")
    args = p.parse_args()
    main(out=args.out, limit=args.limit)
