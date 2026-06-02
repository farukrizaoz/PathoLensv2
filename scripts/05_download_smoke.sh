#!/usr/bin/env bash
# Download the smoke-set TCGA-BRCA DX slides via the GDC open-access API.
# Saves each SVS as {slide_id}.svs so embeddings/masks/captions key on the barcode.
#
# Usage:  bash scripts/05_download_smoke.sh
set -euo pipefail

MANIFEST="${MANIFEST:-data/metadata/smoke_manifest.tsv}"
DATA_DIR="${DATA_DIR:-data/raw/tcga_brca}"
GDC_API="https://api.gdc.cancer.gov/data"

mkdir -p "$DATA_DIR"

if [ ! -f "$MANIFEST" ]; then
    echo "Manifest not found: $MANIFEST"; exit 1
fi

N=$(tail -n +2 "$MANIFEST" | wc -l)
echo "Downloading $N smoke slides -> $DATA_DIR"

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r slide_id file_id file_name mag has_caption; do
    dest="$DATA_DIR/${slide_id}.svs"
    if [ -f "$dest" ]; then
        echo "  SKIP $slide_id"
        continue
    fi
    echo "  -> $slide_id (file_id=$file_id)"
    curl -L --retry 3 --retry-delay 5 -o "${dest}.tmp" "${GDC_API}/${file_id}"
    mv "${dest}.tmp" "$dest"
    echo "  OK $(du -h "$dest" | cut -f1)"
done

echo "Done. Slides:"
ls -lh "$DATA_DIR"
