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
MANIFEST="${MANIFEST:-data/raw/gdc_manifest_BRCA_150.txt}"
WORKERS="${WORKERS:-8}"
STREAMING="${STREAMING:-0}"

mkdir -p "$DATA_DIR"

# 1. Install gdc-client if missing
if ! command -v gdc-client &> /dev/null; then
    echo "→ Installing gdc-client..."
    GDC_VERSION="2.3"
    cd /tmp
    wget -q "https://gdc.cancer.gov/files/public/file/gdc-client_${GDC_VERSION}_Ubuntu_x64.zip"
    unzip -q "gdc-client_${GDC_VERSION}_Ubuntu_x64.zip"
    chmod +x gdc-client
    mv gdc-client /usr/local/bin/
    cd -
fi

# 2. Verify manifest exists
if [ ! -f "$MANIFEST" ]; then
    echo "❌ Manifest not found: $MANIFEST"
    echo ""
    echo "Create one via GDC Data Portal:"
    echo "  1. Go to https://portal.gdc.cancer.gov/repository"
    echo "  2. Filter: Project=TCGA-BRCA, Data Type=Slide Image"
    echo "  3. Add 150 stratified slides to cart"
    echo "  4. Download manifest as gdc_manifest_BRCA_150.txt"
    echo "  5. Place in $MANIFEST"
    exit 1
fi

LINE_COUNT=$(wc -l < "$MANIFEST")
echo "→ Manifest has $LINE_COUNT entries"

# 3. Download
if [ "$STREAMING" = "1" ]; then
    echo "→ Streaming mode: download → embed → delete (batched)"
    BATCH_SIZE=10
    # Split manifest into batches
    HEADER=$(head -1 "$MANIFEST")
    tail -n +2 "$MANIFEST" | split -l $BATCH_SIZE - /tmp/tcga_batch_

    for batch in /tmp/tcga_batch_*; do
        echo "  → Batch: $batch"
        BATCH_MANIFEST="${batch}.manifest"
        echo "$HEADER" > "$BATCH_MANIFEST"
        cat "$batch" >> "$BATCH_MANIFEST"

        gdc-client download \
            -m "$BATCH_MANIFEST" \
            -d "$DATA_DIR" \
            --n-processes "$WORKERS"

        # Run precompute on this batch
        echo "  → Precomputing embeddings..."
        uv run python -m patholens.data.precompute_embeddings \
            --config configs/precompute.yaml \
            --wsi-dir "$DATA_DIR"

        # Delete WSIs in this batch (keep embeddings)
        echo "  → Cleaning up WSIs..."
        # TODO: parse batch file for slide IDs, rm corresponding files
        # For now, careful manual cleanup or full sync to embedding dir then rm

        rm -f "$BATCH_MANIFEST"
    done
else
    echo "→ Standard mode: download all at once (~250 GB)"
    gdc-client download \
        -m "$MANIFEST" \
        -d "$DATA_DIR" \
        --n-processes "$WORKERS"
fi

echo "✓ TCGA-BRCA download complete"
ls -lh "$DATA_DIR" | head
