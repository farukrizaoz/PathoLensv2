#!/usr/bin/env bash
# Re-train the smoke run with the post-audit fixes (faithfulness, grounding GT,
# dtype/device) and run the full eval triad: BLEU/ROUGE + concentration metric.
# Writes results/<run>_caption.json and results/<run>_grounding.json.

set -u
cd /workspace/patholens-vlm
mkdir -p /workspace/logs results
LOG=/workspace/logs/retrain.log
exec >>"$LOG" 2>&1
ts() { date "+%Y-%m-%d %H:%M:%S"; }
fail() { echo "[$(ts)] FAILED at: $*"; touch /workspace/logs/RETRAIN_FAILED; exit 1; }

source .env 2>/dev/null || true
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" WANDB_API_KEY

echo
echo "================ [$(ts)] Dry-run validator ================"
uv run python scripts/dry_run_pipeline.py --config configs/grounding_smoke.yaml \
    > /workspace/logs/dryrun.log 2>&1 || fail "dry-run"

echo
echo "================ [$(ts)] Train (post-audit fixes) ================"
RUN_NAME=grounded_smoke_v2_$(date +%Y%m%d_%H%M%S)
uv run python -m patholens.training.train \
    --config configs/grounding_smoke.yaml \
    --run-name "$RUN_NAME" > /workspace/logs/train.log 2>&1 || fail "train"

CKPT_DIR="checkpoints/${RUN_NAME}/final"
[ -d "$CKPT_DIR" ] || fail "no checkpoint at $CKPT_DIR"
echo "[$(ts)] checkpoint dir: $CKPT_DIR"

echo
echo "================ [$(ts)] Caption metrics (BLEU/ROUGE) ================"
uv run python -m patholens.evaluation.caption_metrics \
    --checkpoint "$CKPT_DIR" \
    --test-set data/processed/slideinstruction_smoke/test.json \
    --embeddings-dir data/processed/embeddings/tcga_brca \
    --output-json "results/${RUN_NAME}_caption.json" \
    --max-vision-tokens 1024 > /workspace/logs/eval_caption.log 2>&1 || fail "caption eval"

echo
echo "================ [$(ts)] Concentration metric ================"
uv run python -m patholens.evaluation.concentration_metric \
    --checkpoint "$CKPT_DIR" \
    --test-set data/processed/slideinstruction_smoke/test.json \
    --embeddings-dir data/processed/embeddings/tcga_brca \
    --output-json "results/${RUN_NAME}_grounding.json" \
    --max-vision-tokens 1024 > /workspace/logs/eval_grounding.log 2>&1 || fail "concentration eval"

echo
echo "================ [$(ts)] DONE ================"
echo "run_name: $RUN_NAME"
echo "checkpoint: $CKPT_DIR"
echo "caption_metrics: results/${RUN_NAME}_caption.json"
echo "grounding_metrics: results/${RUN_NAME}_grounding.json"
touch /workspace/logs/RETRAIN_OK
