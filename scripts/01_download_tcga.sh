#!/usr/bin/env bash
# Download 150 TCGA-BRCA WSIs (stratified) using GDC client.
#
# Steps:
#   1. Install gdc-client (binary, not pip package)
#   2. Use existing manifest data/raw/gdc_manifest_BRCA_150.txt
#      (manifest must be created beforehand via GDC Portal — see docs/DATA.md)
#   3. Parallel download with 8 workers
#   4. Optional streaming mode: download N at a time, embed, delete
#
# Usage:
#   make download-tcga            # standard
#   STREAMING=1 make download-tcga  # streaming mode (saves disk)

set -euo pipefail

DATA_DIR="${DATA_DIR:-data/raw/tcga_brca}"
MANIFEST="${MANIFEST:-data/metadata/gdc_manifest_bcss151.txt}"
STREAMING="${STREAMING:-0}"
GDC_API="https://api.gdc.cancer.gov/data"

mkdir -p "$DATA_DIR"

# 1. Verify manifest exists
if [ ! -f "$MANIFEST" ]; then
    echo "❌ Manifest not found: $MANIFEST"
    exit 1
fi

LINE_COUNT=$(tail -n +2 "$MANIFEST" | wc -l)
echo "→ Manifest has $LINE_COUNT slides"

download_slide() {
    local file_id="$1"
    local filename="$2"
    local dest="$DATA_DIR/$filename"

    if [ -f "$dest" ]; then
        echo "  SKIP $filename (already exists)"
        return 0
    fi

    echo "  → Downloading $filename..."
    curl -L --retry 3 --retry-delay 5 -o "${dest}.tmp" \
        "${GDC_API}/${file_id}"
    mv "${dest}.tmp" "$dest"
    echo "  ✓ $filename"
}

precompute_and_clean() {
    echo "  → Precomputing embeddings..."
    uv run python -m patholens.data.precompute_embeddings \
        --config configs/precompute.yaml \
        --wsi-dir "$DATA_DIR"

    echo "  → Cleaning up WSIs to save disk..."
    find "$DATA_DIR" -name "*.svs" -delete
    find "$DATA_DIR" -name "*.tif" -delete
    echo "  ✓ WSIs deleted, embeddings kept"
}

if [ "$STREAMING" = "1" ]; then
    echo "→ Streaming mode: download → embed → delete (slide by slide)"
    tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r file_id filename md5 size state; do
        download_slide "$file_id" "$filename"
        precompute_and_clean
    done
else
    echo "→ Standard mode: download all slides (~86 GB)"
    tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r file_id filename md5 size state; do
        download_slide "$file_id" "$filename"
    done
fi

echo "✓ TCGA-BRCA download complete"
ls -lh "$DATA_DIR" | head
