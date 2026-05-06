.PHONY: help install install-dev install-gpu test lint format clean
.PHONY: sync-up sync-down ssh gpu-info
.PHONY: download-tcga download-camelyon prepare-instruction precompute
.PHONY: train-baseline train-grounded train-grounded-v2
.PHONY: eval eval-pointing eval-intervention eval-caption
.PHONY: notebook tensorboard
.DEFAULT_GOAL := help

# Load .env
-include .env
export

# Defaults (override in .env)
RUNPOD_HOST ?= root@xxx.proxy.runpod.net
RUNPOD_PORT ?= 22
REMOTE_DIR ?= /workspace/patholens-vlm

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ============================================================
# Setup
# ============================================================

install:  ## Install package (lokal: CPU/Mac safe)
	uv sync

install-dev:  ## Install with dev tools + pre-commit
	uv sync --extra dev
	uv run pre-commit install

install-gpu:  ## Install GPU extras (RunPod-only, includes bitsandbytes)
	uv sync --extra dev --extra gpu

clean:  ## Clean caches and artifacts
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

# ============================================================
# Quality checks
# ============================================================

test:  ## Run smoke tests (no GPU, no data)
	uv run pytest tests/ -v -m "not slow and not gpu"

test-all:  ## Run all tests including slow/GPU
	uv run pytest tests/ -v

lint:  ## Ruff check + format check
	uv run ruff check src/ tests/ scripts/
	uv run ruff format --check src/ tests/ scripts/

format:  ## Auto-format with ruff
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

# ============================================================
# Sync (local <-> RunPod)
# ============================================================

sync-up:  ## rsync local code → RunPod (excludes data/checkpoints)
	@echo "→ Syncing to $(RUNPOD_HOST):$(REMOTE_DIR)"
	rsync -avz --progress \
		--exclude='.git/' \
		--exclude='data/' \
		--exclude='checkpoints/' \
		--exclude='wandb/' \
		--exclude='__pycache__/' \
		--exclude='.venv/' \
		--exclude='.pytest_cache/' \
		--exclude='.ruff_cache/' \
		--exclude='node_modules/' \
		-e "ssh -p $(RUNPOD_PORT)" \
		./ $(RUNPOD_HOST):$(REMOTE_DIR)/

sync-down:  ## rsync RunPod outputs → local (checkpoints, wandb, embeddings)
	@echo "← Syncing FROM $(RUNPOD_HOST):$(REMOTE_DIR)"
	mkdir -p checkpoints wandb data/processed
	rsync -avz --progress \
		-e "ssh -p $(RUNPOD_PORT)" \
		$(RUNPOD_HOST):$(REMOTE_DIR)/checkpoints/ ./checkpoints/ || true
	rsync -avz --progress \
		-e "ssh -p $(RUNPOD_PORT)" \
		$(RUNPOD_HOST):$(REMOTE_DIR)/wandb/ ./wandb/ || true
	rsync -avz --progress \
		-e "ssh -p $(RUNPOD_PORT)" \
		$(RUNPOD_HOST):$(REMOTE_DIR)/data/processed/ ./data/processed/ || true

ssh:  ## SSH into RunPod
	ssh -p $(RUNPOD_PORT) $(RUNPOD_HOST)

gpu-info:  ## Print GPU info (run on RunPod to verify)
	@nvidia-smi || echo "No nvidia-smi (probably local)"
	@uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB' if torch.cuda.is_available() else 'No CUDA')"

# ============================================================
# Data
# ============================================================

download-tcga:  ## Download 150 TCGA-BRCA slides (RunPod, ~12-24h)
	bash scripts/01_download_tcga.sh

download-camelyon:  ## Download CAMELYON16 test subset (~150 GB)
	bash scripts/02_download_camelyon.sh

prepare-instruction:  ## Download SlideInstruction from HuggingFace
	uv run python scripts/04_prepare_slideinstruction.py

precompute:  ## CONCHv1.5 + TITAN embedding precompute (RunPod, ~10h)
	uv run python -m patholens.data.precompute_embeddings \
		--config configs/precompute.yaml

# ============================================================
# Training
# ============================================================

train-baseline:  ## Caption-only baseline (no grounding loss)
	uv run python -m patholens.training.train \
		--config configs/caption_baseline.yaml \
		--run-name baseline_$(shell date +%Y%m%d_%H%M%S)

train-grounded:  ## Grounding loss v1 (CONCH pseudo-supervision)
	uv run python -m patholens.training.train \
		--config configs/grounding_v1.yaml \
		--run-name grounded_v1_$(shell date +%Y%m%d_%H%M%S)

train-grounded-v2:  ## Grounding loss v2 (CAMELYON-supervised + faithfulness reg)
	uv run python -m patholens.training.train \
		--config configs/grounding_v2.yaml \
		--run-name grounded_v2_$(shell date +%Y%m%d_%H%M%S)

# ============================================================
# Evaluation
# ============================================================

eval:  ## Run all evals (pointing + intervention + caption)
	uv run python -m patholens.evaluation.run_all \
		--checkpoint checkpoints/latest \
		--output-dir results/$(shell date +%Y%m%d_%H%M%S)/

eval-pointing:  ## CAMELYON16 pointing game only
	uv run python -m patholens.evaluation.pointing_game \
		--checkpoint checkpoints/latest \
		--camelyon-dir data/processed/embeddings/camelyon16

eval-intervention:  ## Faithfulness intervention test
	uv run python -m patholens.evaluation.intervention_test \
		--checkpoint checkpoints/latest

eval-caption:  ## BLEU/ROUGE/METEOR vs ground-truth reports
	uv run python -m patholens.evaluation.caption_metrics \
		--checkpoint checkpoints/latest \
		--test-set data/processed/slideinstruction/test.json

# ============================================================
# Misc
# ============================================================

notebook:  ## Launch JupyterLab
	uv run jupyter lab

tensorboard:  ## Tensorboard for wandb local logs
	uv run tensorboard --logdir wandb/
