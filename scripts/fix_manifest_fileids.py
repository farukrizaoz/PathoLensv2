"""Patch smoke_manifest.tsv with current GDC file_ids (BCSS-embedded UUIDs are stale).

Queries the GDC files API for all open-access TCGA-BRCA SVS slides, maps each
barcode to its current file_id, and rewrites the manifest's file_id/file_name.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

BAR = re.compile(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+-\d+[A-Z]-\d+-DX\d+)")
MANIFEST = Path("data/metadata/smoke_manifest.tsv")


def query(page: int, size: int = 100) -> dict:
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
                {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
                {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
                {"op": "in", "content": {"field": "access", "value": ["open"]}},
            ],
        },
        "fields": "file_id,file_name,file_size",
        "size": size,
        "from": page * size,
    }
    req = urllib.request.Request(
        "https://api.gdc.cancer.gov/files",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def main() -> None:
    mapping: dict[str, dict] = {}
    page = 0
    total = None
    while True:
        r = query(page)
        if total is None:
            total = r["data"]["pagination"]["total"]
        hits = r["data"]["hits"]
        if not hits:
            break
        for h in hits:
            m = BAR.search(h["file_name"])
            if m:
                mapping[m.group(1)] = {"fid": h["file_id"], "fn": h["file_name"], "sz": h.get("file_size", 0)}
        page += 1
        if page * 100 >= total:
            break
        time.sleep(0.2)
    print(f"GDC barcodes: {len(mapping)}")

    lines = MANIFEST.read_text().strip().split("\n")
    out = [lines[0]]
    missing = []
    for ln in lines[1:]:
        sid = ln.split("\t")[0]
        g = mapping.get(sid)
        if g:
            out.append(f"{sid}\t{g['fid']}\t{g['fn']}\t40.0\t1")
        else:
            missing.append(sid)
    MANIFEST.write_text("\n".join(out) + "\n")
    print(f"patched: {len(out) - 1}  missing: {missing}")


if __name__ == "__main__":
    main()
