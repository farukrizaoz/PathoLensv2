"""Convert raw SlideChat conversations into PathoLens dataset format.

Raw SlideChat entry:
    {"id", "image": ["./BRCA/TCGA-..-DX1.csv"], "conversations": [{from,value}, ...]}

PathoLens GroundingSlideDataset expects per-slide examples:
    {"slide_id", "caption", "instruction", "vqa_pairs": [{question, answer}, ...]}

Strategy:
  - Keep BRCA only (image path contains /BRCA/).
  - Group all conversation pairs by slide barcode (across raw train/val/test).
  - Choose the most "report-like" gpt answer as the caption (prompt mentions
    describe / report / summary / findings / overview); fallback = longest answer.
  - Remaining (human, gpt) pairs become vqa_pairs.
  - Split slides 70/15/15 deterministically (a slide lives in exactly one split).

The GroundingSlideDataset filters by embedding-h5 existence, so converting all
881 BRCA slides is fine — only slides we actually embedded get used.

Usage:
    python scripts/07_convert_slidechat.py \
        --raw-dir data/processed/slideinstruction \
        --out-dir data/processed/slideinstruction_brca
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

BARCODE_RE = re.compile(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+-\d+[A-Z]-\d+-DX\d+)")
REPORT_KEYWORDS = (
    "report",
    "describe",
    "description",
    "summary",
    "summarize",
    "findings",
    "overview",
    "explanation",
    "explain",
    "pivotal",
    "key features",
    "diagnosis",
)


def _is_report_prompt(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in REPORT_KEYWORDS)


def _load_raw(raw_dir: Path) -> list[dict]:
    examples: list[dict] = []
    for split in ("train", "val", "test"):
        fp = raw_dir / f"{split}.json"
        if fp.exists():
            examples.extend(json.load(open(fp)))
    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/processed/slideinstruction")
    ap.add_argument("--out-dir", default="data/processed/slideinstruction_brca")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = _load_raw(Path(args.raw_dir))
    print(f"Loaded {len(raw)} raw SlideChat examples")

    # Group conversation pairs by slide barcode (BRCA only)
    by_slide: dict[str, list[tuple[str, str]]] = {}
    for ex in raw:
        img = ex["image"][0] if isinstance(ex["image"], list) else ex["image"]
        if "/BRCA/" not in img:
            continue
        m = BARCODE_RE.search(img)
        if not m:
            continue
        barcode = m.group(1)
        convo = ex["conversations"]
        # pair consecutive human/gpt turns
        for i in range(0, len(convo) - 1, 2):
            if convo[i]["from"] == "human" and convo[i + 1]["from"] == "gpt":
                q = convo[i]["value"].replace("<image>", "").strip()
                a = convo[i + 1]["value"].strip()
                by_slide.setdefault(barcode, []).append((q, a))

    print(f"BRCA slides: {len(by_slide)}")

    # Build per-slide examples
    examples: list[dict] = []
    for slide_id, pairs in by_slide.items():
        # pick caption = report-like answer, else longest
        report_pairs = [(q, a) for q, a in pairs if _is_report_prompt(q)]
        if report_pairs:
            cap_q, caption = max(report_pairs, key=lambda qa: len(qa[1]))
        else:
            cap_q, caption = max(pairs, key=lambda qa: len(qa[1]))
        vqa = [{"question": q, "answer": a} for q, a in pairs if (q, a) != (cap_q, caption)]
        examples.append(
            {
                "slide_id": slide_id,
                "caption": caption,
                "instruction": cap_q
                or "Generate a structured pathology report for this whole-slide image.",
                "vqa_pairs": vqa,
            }
        )

    # Deterministic 70/15/15 split by slide
    random.seed(args.seed)
    random.shuffle(examples)
    n = len(examples)
    n_tr = int(n * 0.70)
    n_val = int(n * 0.15)
    splits = {
        "train": examples[:n_tr],
        "val": examples[n_tr : n_tr + n_val],
        "test": examples[n_tr + n_val :],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, exs in splits.items():
        fp = out_dir / f"{name}.json"
        json.dump(exs, open(fp, "w"), indent=2)
        print(f"  {name}: {len(exs)} slides -> {fp}")

    # quick caption length stats
    lens = [len(e["caption"]) for e in examples]
    print(f"caption chars: min={min(lens)} mean={sum(lens)//len(lens)} max={max(lens)}")


if __name__ == "__main__":
    main()
