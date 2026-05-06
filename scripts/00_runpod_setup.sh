#!/usr/bin/env bash
# RunPod first-time setup script.
# Run after first SSH into a fresh A6000 pod.
#
# Usage:
#   ssh into pod, then:
#     cd /workspace/patholens-vlm
#     bash scripts/00_runpod_setup.sh

set -euo pipefail

echo "=== PathoLens-VLM RunPod Setup ==="

# 1. System libraries (openslide for WSI)
echo "→ Installing system libraries..."
apt-get update -qq
apt-get install -y -qq \
    openslide-tools \
    libopenslide0 \
    python3-openslide \
    libgl1 \
    libglib2.0-0 \
    tmux \
    htop \
    nvtop \
    rsync

# 2. Install uv (Python package manager)
echo "→ Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installs to ~/.cargo/bin or similar
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi
uv --version

# 3. Verify GPU
echo "→ GPU info:"
nvidia-smi
echo ""
echo "→ Verifying CUDA in PyTorch..."

# Install package with GPU extras
echo "→ Installing patholens-vlm[gpu, dev]..."
uv sync --extra gpu --extra dev

# 4. Verify torch + CUDA
uv run python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA {torch.version.cuda}')
print(f'✓ Device: {torch.cuda.get_device_name(0)}')
print(f'✓ VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

# 5. HuggingFace login (token from .env)
echo "→ HuggingFace login..."
if [ -f .env ]; then
    set -a; source .env; set +a
    if [ -n "${HF_TOKEN:-}" ]; then
        uv run huggingface-cli login --token "$HF_TOKEN"
    else
        echo "⚠ HF_TOKEN not set in .env, skipping HF login"
    fi
else
    echo "⚠ .env not found — copy .env.example to .env and fill in tokens"
fi

# 6. WandB login
if [ -n "${WANDB_API_KEY:-}" ]; then
    uv run wandb login --relogin "$WANDB_API_KEY"
else
    echo "⚠ WANDB_API_KEY not set, skipping wandb login"
fi

# 7. Create directory structure
mkdir -p data/raw/{tcga_brca,camelyon16}
mkdir -p data/processed/{embeddings,tissue_masks,camelyon_patch_masks,slideinstruction}
mkdir -p data/metadata
mkdir -p checkpoints
mkdir -p wandb

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. make download-tcga       # ~12-24h, run in tmux"
echo "  2. make download-camelyon   # ~6-8h, run in tmux"
echo "  3. make prepare-instruction"
echo "  4. make precompute          # ~10h, run in tmux"
echo ""
echo "Always use tmux for long-running tasks:"
echo "  tmux new -s download"
echo "  make download-tcga"
echo "  Ctrl+B, D  # detach"
