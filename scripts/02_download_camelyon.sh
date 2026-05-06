#!/usr/bin/env bash
# Download CAMELYON16 test set (130 slides, ~150 GB).
# Source: AWS S3 mirror (no auth needed, Grand Challenge data).
#
# Usage:
#   make download-camelyon

set -euo pipefail

DATA_DIR="${DATA_DIR:-data/raw/camelyon16}"
TEST_DIR="$DATA_DIR/test"
mkdir -p "$TEST_DIR/images" "$TEST_DIR/annotations"

# Install awscli if missing
if ! command -v aws &> /dev/null; then
    echo "→ Installing awscli..."
    pip install awscli --quiet
fi

# CAMELYON16 is hosted on AWS S3 public bucket
S3_BASE="s3://camelyon-dataset/CAMELYON16"

echo "→ Downloading CAMELYON16 test images (~130 files, ~150 GB)..."
aws s3 cp \
    --no-sign-request \
    --recursive \
    "${S3_BASE}/testing/images/" \
    "$TEST_DIR/images/"

echo "→ Downloading CAMELYON16 test annotations (XML)..."
aws s3 cp \
    --no-sign-request \
    --recursive \
    "${S3_BASE}/testing/lesion_annotations/" \
    "$TEST_DIR/annotations/"

echo "→ Downloading reference labels..."
aws s3 cp \
    --no-sign-request \
    "${S3_BASE}/testing/reference.csv" \
    "$TEST_DIR/reference.csv"

echo "✓ CAMELYON16 download complete"
echo "  Images: $(ls $TEST_DIR/images | wc -l)"
echo "  Annotations: $(ls $TEST_DIR/annotations | wc -l)"
df -h "$DATA_DIR"
