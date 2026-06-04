#!/usr/bin/env bash
# Full-scale single-shot runner. One pod, one button, every artifact preserved.
#
# Phases (each gated by a flag file so this script is resume-safe):
#   1. download    — fetch SVS files for slides not yet on disk
#   2. precompute  — CONCHv1.5 patch embeddings (skips slides whose H5 exists)
#   3. bcss        — build per-patch BCSS coverage HDF5s
#   4. dry_run     — validate the full data path on the new pool
#   5. train       — 8 epochs LoRA+Adapter fine-tune (configs/fullscale.yaml)
#   6. eval        — BLEU/ROUGE + concentration metric on the 7-slide test set
#   7. pointing    — pointing-game on BCSS held-out (cheap, no extra download)
#   8. report      — emit docs/FULLSCALE_RUN_REPORT.md with all numbers
#
# Logs land in /workspace/logs/fullscale/<phase>.log
# Progress flags in /workspace/logs/fullscale/<phase>.OK
# Final success flag /workspace/logs/fullscale/RUN_OK

set -u
cd /workspace/patholens-vlm
LOG_DIR=/workspace/logs/fullscale
mkdir -p "$LOG_DIR" results

# Stream every line to the master log and per-phase logs.
exec >>"$LOG_DIR/master.log" 2>&1
ts() { date '+%Y-%m-%d %H:%M:%S'; }
section() { echo; echo "================ [$(ts)] $* ================"; }
fail() { echo "[$(ts)] FAILED at: $*"; touch "$LOG_DIR/RUN_FAILED"; exit 1; }
phase_done() { touch "$LOG_DIR/${1}.OK"; }
phase_pending() { [ ! -f "$LOG_DIR/${1}.OK" ]; }

source .env
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" WANDB_API_KEY

RUN_NAME="grounded_fullscale_$(date +%Y%m%d_%H%M%S)"
echo "RUN_NAME=$RUN_NAME"
echo "$RUN_NAME" > "$LOG_DIR/RUN_NAME"

# ── Phase 1: download SVS for slides not on disk ────────────────────────────
if phase_pending download; then
    section "1. Download missing SVS files"
    while IFS=$'\t' read -r sid file_id file_name mag has_caption; do
        [ "$sid" = "slide_id" ] && continue
        dest="data/raw/tcga_brca/${sid}.svs"
        if [ -s "$dest" ]; then
            echo "  SKIP $sid (have $(du -h "$dest" | cut -f1))"
            continue
        fi
        echo "  -> $sid (file_id=$file_id)"
        curl -L --retry 3 --retry-delay 5 -o "${dest}.tmp" \
            "https://api.gdc.cancer.gov/data/${file_id}" >> "$LOG_DIR/download.log" 2>&1 || fail "download $sid"
        size=$(stat -c %s "${dest}.tmp")
        if [ "$size" -lt 1000000 ]; then
            head -c 200 "${dest}.tmp"; echo
            rm -f "${dest}.tmp"
            fail "download $sid produced only $size bytes (likely error JSON)"
        fi
        mv "${dest}.tmp" "$dest"
        echo "  OK $sid ($(du -h "$dest" | cut -f1))"
    done < data/metadata/fullscale_manifest.tsv
    phase_done download
fi

# ── Phase 2: precompute embeddings ──────────────────────────────────────────
if phase_pending precompute; then
    section "2. Precompute CONCHv1.5 patch embeddings (resume-safe)"
    uv run python -m patholens.data.precompute_embeddings \
        --config configs/precompute.yaml \
        --wsi-dir data/raw/tcga_brca \
        --resume true >> "$LOG_DIR/precompute.log" 2>&1 || fail "precompute"
    n_h5=$(ls data/processed/embeddings/tcga_brca/*.h5 2>/dev/null | wc -l)
    echo "  H5s on disk: $n_h5"
    [ "$n_h5" -ge 50 ] || fail "expected >=50 H5s, have $n_h5"
    phase_done precompute
fi

# ── Phase 3: BCSS patch masks for new slides ────────────────────────────────
if phase_pending bcss; then
    section "3. Download + build BCSS patch masks"
    # Download masks listed in fullscale_bcss_rois (skip if PNG already present)
    uv run python - <<'PY' >> "$LOG_DIR/bcss.log" 2>&1
import csv, urllib.request
from pathlib import Path
RAW = Path("data/raw/bcss_masks"); RAW.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open("data/metadata/fullscale_bcss_rois.tsv"), delimiter="\t"))
print(f"ROIs to ensure: {len(rows)}")
for r in rows:
    fname = f"{r['slide_id']}_xmin{r['xmin']}_ymin{r['ymin']}_MPP-0.25.png"
    dest = RAW / fname
    if dest.exists() and dest.stat().st_size > 0:
        continue
    print(f"  -> {fname}")
    try:
        urllib.request.urlretrieve(r["mask_link"], dest)
    except Exception as e:
        print(f"  FAILED: {e}")
print(f"PNGs on disk: {len(list(RAW.glob('*.png')))}")
PY
    uv run python scripts/06_prepare_bcss_masks.py build >> "$LOG_DIR/bcss.log" 2>&1 || fail "bcss build"
    n_masks=$(ls data/processed/bcss_patch_masks/*.h5 2>/dev/null | wc -l)
    echo "  BCSS patch-mask H5s: $n_masks"
    [ "$n_masks" -ge 50 ] || fail "expected >=50 BCSS masks, have $n_masks"
    phase_done bcss
fi

# ── Phase 4: dry-run validator ──────────────────────────────────────────────
if phase_pending dry_run; then
    section "4. Dry-run validator on the new pool"
    uv run python scripts/dry_run_pipeline.py \
        --config configs/fullscale.yaml >> "$LOG_DIR/dry_run.log" 2>&1 || fail "dry_run"
    phase_done dry_run
fi

# ── Phase 5: training ───────────────────────────────────────────────────────
if phase_pending train; then
    section "5. Train (8 epochs, retuned losses + warmup)"
    uv run python -m patholens.training.train \
        --config configs/fullscale.yaml \
        --run-name "$RUN_NAME" >> "$LOG_DIR/train.log" 2>&1 || fail "train"
    CKPT_DIR="checkpoints/${RUN_NAME}/final"
    [ -d "$CKPT_DIR" ] || fail "no checkpoint at $CKPT_DIR"
    echo "$CKPT_DIR" > "$LOG_DIR/CKPT_DIR"
    phase_done train
fi

CKPT_DIR=$(cat "$LOG_DIR/CKPT_DIR")

# ── Phase 6: BLEU/ROUGE + concentration eval ────────────────────────────────
if phase_pending eval; then
    section "6a. Caption metrics (BLEU/ROUGE)"
    uv run python -m patholens.evaluation.caption_metrics \
        --checkpoint "$CKPT_DIR" \
        --test-set data/processed/slideinstruction_fullscale/test.json \
        --embeddings-dir data/processed/embeddings/tcga_brca \
        --output-json "results/${RUN_NAME}_caption.json" \
        --max-vision-tokens 1024 >> "$LOG_DIR/eval_caption.log" 2>&1 || fail "caption eval"

    section "6b. Concentration metric"
    uv run python -m patholens.evaluation.concentration_metric \
        --checkpoint "$CKPT_DIR" \
        --test-set data/processed/slideinstruction_fullscale/test.json \
        --embeddings-dir data/processed/embeddings/tcga_brca \
        --output-json "results/${RUN_NAME}_grounding.json" \
        --max-vision-tokens 1024 >> "$LOG_DIR/eval_grounding.log" 2>&1 || fail "concentration eval"
    phase_done eval
fi

# ── Phase 7: BCSS pointing-game (uses our own BCSS masks as ground truth) ──
if phase_pending pointing; then
    section "7. BCSS pointing-game on held-out test set"
    uv run python -m patholens.evaluation.bcss_pointing_game \
        --checkpoint "$CKPT_DIR" \
        --test-set data/processed/slideinstruction_fullscale/test.json \
        --embeddings-dir data/processed/embeddings/tcga_brca \
        --bcss-masks-dir data/processed/bcss_patch_masks \
        --output-json "results/${RUN_NAME}_pointing.json" \
        --max-vision-tokens 1024 >> "$LOG_DIR/pointing.log" 2>&1 || fail "pointing"
    phase_done pointing
fi

# ── Phase 8: final report ───────────────────────────────────────────────────
if phase_pending report; then
    section "8. Final report"
    uv run python scripts/fullscale_writeup.py \
        --run-name "$RUN_NAME" \
        --output docs/FULLSCALE_RUN_REPORT.md >> "$LOG_DIR/report.log" 2>&1 || fail "report"
    phase_done report
fi

touch "$LOG_DIR/RUN_OK"
section "ALL PHASES OK"
echo "Checkpoint: $CKPT_DIR"
echo "Report:     docs/FULLSCALE_RUN_REPORT.md"
echo "Logs:       $LOG_DIR/"
echo "Results:    results/${RUN_NAME}_{caption,grounding,pointing}.json"
