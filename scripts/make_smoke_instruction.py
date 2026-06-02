"""Build a tiny instruction set restricted to the smoke-manifest slides.

Pulls the converted per-slide BRCA examples (from 07_convert_slidechat.py),
keeps only the slides in smoke_manifest.tsv, and writes an 8/2 train/val split
so the smoke training run has guaranteed, matched data.

Usage:
    python scripts/make_smoke_instruction.py
"""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST = Path("data/metadata/smoke_manifest.tsv")
BRCA_DIR = Path("data/processed/slideinstruction_brca")
OUT_DIR = Path("data/processed/slideinstruction_smoke")


def main() -> None:
    smoke_ids = [ln.split("\t")[0] for ln in MANIFEST.read_text().strip().split("\n")[1:]]
    smoke_ids_set = set(smoke_ids)

    pool: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        fp = BRCA_DIR / f"{split}.json"
        if fp.exists():
            for ex in json.load(open(fp)):
                if ex["slide_id"] in smoke_ids_set:
                    pool[ex["slide_id"]] = ex

    ordered = [pool[s] for s in smoke_ids if s in pool]
    print(f"Matched {len(ordered)}/{len(smoke_ids)} smoke slides to captions")

    n_val = max(1, len(ordered) // 5)  # ~20% val
    val = ordered[:n_val]
    train = ordered[n_val:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(train, open(OUT_DIR / "train.json", "w"), indent=2)
    json.dump(val, open(OUT_DIR / "val.json", "w"), indent=2)
    json.dump(ordered, open(OUT_DIR / "test.json", "w"), indent=2)  # all 10 for eval
    print(f"  train={len(train)}  val={len(val)}  test(all)={len(ordered)} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
