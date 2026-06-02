#!/usr/bin/env bash
# Overnight smoke chain: wait for precompute → build masks → make smoke set
# → train → caption-metrics eval → halt pod. Each step logs to its own file
# and the chain stops on first failure. Final action is `poweroff` so RunPod
# bills only storage afterwards (pod state preserved).

set -u
cd /workspace/patholens-vlm
mkdir -p /workspace/logs

LOG=/workspace/logs/overnight.log
exec >>"$LOG" 2>&1

ts() { date "+%Y-%m-%d %H:%M:%S"; }
step() { echo; echo "================ [$(ts)] $* ================"; }
fail() { echo "[$(ts)] FAILED at: $*"; touch /workspace/logs/OVERNIGHT_FAILED; poweroff; exit 1; }

step "Waiting for precompute (pid=$(pgrep -f precompute_embed | head -1))"
while pgrep -f precompute_embed >/dev/null; do sleep 60; done
n_h5=$(ls data/processed/embeddings/tcga_brca/*.h5 2>/dev/null | wc -l)
echo "[$(ts)] precompute exited — $n_h5/10 H5 files present"
[ "$n_h5" -ge 1 ] || fail "no embeddings produced"

source .env 2>/dev/null || true
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" WANDB_API_KEY

step "Build BCSS patch masks"
uv run python scripts/06_prepare_bcss_masks.py build > /workspace/logs/bcss_build.log 2>&1 || fail "bcss build"
n_masks=$(ls data/processed/bcss_patch_masks/*.h5 2>/dev/null | wc -l)
echo "[$(ts)] bcss_patch_masks built: $n_masks"

step "Build smoke instruction set"
uv run python scripts/make_smoke_instruction.py > /workspace/logs/smoke_instr.log 2>&1 || fail "smoke instr"

step "Train smoke"
RUN_NAME=grounded_smoke_$(date +%Y%m%d_%H%M%S)
uv run python -m patholens.training.train \
    --config configs/grounding_smoke.yaml \
    --run-name "$RUN_NAME" > /workspace/logs/train.log 2>&1 || fail "train"

# Determine checkpoint dir written by save_pretrained
CKPT_DIR=$(ls -dt checkpoints/${RUN_NAME}* 2>/dev/null | head -1)
echo "[$(ts)] checkpoint dir: $CKPT_DIR"
[ -n "$CKPT_DIR" ] || fail "checkpoint dir not found"

step "Caption-metrics eval"
mkdir -p results
uv run python -m patholens.evaluation.caption_metrics \
    --checkpoint "$CKPT_DIR" \
    --test-set data/processed/slideinstruction_smoke/test.json \
    --embeddings-dir data/processed/embeddings/tcga_brca \
    --output-json results/${RUN_NAME}_caption.json \
    --max-vision-tokens 1024 \
    > /workspace/logs/eval.log 2>&1 || fail "eval"

step "DONE — halting pod"
touch /workspace/logs/OVERNIGHT_OK
sync
sleep 5
poweroff
