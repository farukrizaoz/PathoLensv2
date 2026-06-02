#!/usr/bin/env bash
# Launch the PathoLens-VLM Streamlit demo (run on the GPU host).
#
# Tunnel from your laptop in a separate terminal:
#   ssh -L 8501:localhost:8501 root@<host> -p <port> -i ~/.ssh/id_ed25519 -N
# then open http://localhost:8501

set -u
cd /workspace/patholens-vlm
[ -f .env ] && source .env
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" WANDB_API_KEY

exec uv run streamlit run src/patholens/app/streamlit_app.py \
    --server.port "${STREAMLIT_PORT:-8501}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
