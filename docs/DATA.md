# Data

Data sources, download instructions, preprocessing pipeline, and storage layout.

## Overview

| Dataset | Purpose | Raw Size | Processed | Location |
|---------|---------|----------|-----------|----------|
| TCGA-BRCA (151 BCSS-overlap slides) | Primary training | ~255 GB | ~22 GB HDF5 | RunPod (raw), local (processed) |
| BCSS (151 ROI annotations) | Per-class grounding GT | ~3 GB ROI PNGs | ~2 GB HDF5 | RunPod → local |
| CAMELYON16 (~100 test slides) | Pointing game eval only | ~150 GB | ~10 GB HDF5 | RunPod |
| SlideInstruction | Instruction captions | ~5 GB JSON | ~5 GB | Local + RunPod |

**Storage discipline:** Raw WSI files stay on RunPod only. Only HDF5 embeddings and metadata are synced to local (~30 GB).

---

## TCGA-BRCA: BCSS-Overlap Cohort (151 Slides)

### Slide Selection

The training cohort is the 151 TCGA-BRCA slides for which BCSS pixel-level ROI annotations exist. This enables the hybrid BCSS + pseudo-GT grounding supervision.

> This replaces the earlier stratified 150-slide selection plan. BCSS annotation availability is the binding constraint — we use all 151 annotated slides rather than imposing an arbitrary stratification.

**Approximate subtype distribution (subject to BCSS composition):**
- IDC (Invasive Ductal Carcinoma): ~90 slides
- ILC (Invasive Lobular Carcinoma): ~20 slides
- Other subtypes: ~41 slides

The BCSS slide list is the canonical source (`data/metadata/bcss_slide_ids.txt`, generated during BCSS preprocessing).

### Download

```bash
# Run on RunPod — ~12–24 hours
make download-tcga
```

The download script (`scripts/01_download_tcga.sh`) reads `data/metadata/bcss_slide_ids.txt` to build the GDC manifest and downloads only those 151 slides.

### Clinical Labels

```
data/metadata/tcga_brca_clinical.tsv
```

Key fields from GDC clinical export: `case_id`, `histological_type`, `tumor_grade`, `pathologic_stage`, `er_status`, `pr_status`, `her2_status`.

### Pathology Reports

**We do NOT parse raw TCGA PDF reports.** SlideInstruction provides GPT-4 structured captions for all 151 slides. These are the training targets.

---

## BCSS (Breast Cancer Semantic Segmentation)

### Description

- **151 TCGA-BRCA slides** with pixel-level ROI annotation (same SVS files as our training set)
- **22 tissue classes** → 4 primary classes used:
  - `tumor` (class 1 in BCSS)
  - `stroma` (class 2)
  - `lymphocytic_infiltrate` → `lymph` (class 3)
  - `necrosis_or_debris` → `necrosis` (class 12)
- ROI-only annotations (not full-slide) — typically 20–60% patch coverage
- **CC0 license** (public domain, no restrictions)
- Source: Grand Challenge Girder server (no authentication required)

Reference: Amgad et al. (2019), GigaScience. https://doi.org/10.1093/gigascience/giz037

### Download and Preprocessing

```bash
# Downloads RGB ROI tiles + label-map PNGs, rasterizes to patch grid
uv run python -m patholens.data.bcss_loader \
    --slide-ids data/metadata/bcss_slide_ids.txt \
    --wsi-dir data/raw/tcga_brca \
    --output-dir data/processed/bcss_patch_masks \
    --patch-size 448 \
    --magnification 20
```

**Pipeline (`src/patholens/data/bcss_loader.py`):**

1. Download RGB ROI tile + label-map PNG from Girder server for each annotated slide
2. Align BCSS annotation space to our 20× patch grid:
   - BCSS native: 40× (0.25 MPP) → downsample 2× to match our 20× patches
   - For each patch, compute fraction of pixels belonging to each class
3. Write per-slide HDF5

### Output HDF5 Schema

```
data/processed/bcss_patch_masks/{slide_id}.h5
├── mask_tumor:    (N_patches,) float32   # fraction of patch pixels = tumor class
├── mask_stroma:   (N_patches,) float32
├── mask_lymph:    (N_patches,) float32
├── mask_necrosis: (N_patches,) float32
└── roi_coverage:  (N_patches,) bool      # True = patch is within annotated ROI
```

`N_patches` matches the patch count in the corresponding embedding HDF5. All arrays are aligned by patch index.

**Patch index alignment:** The embedding precompute step (`precompute_embeddings.py`) and the BCSS loader must use the same tissue-mask-derived patch grid. Both use the same `extract_patches()` function from `wsi_preprocessing.py`. If a BCSS HDF5 is missing for a slide, training falls back to CONCH pseudo-GT for all sentences from that slide.

### Routing Logic in Training

`src/patholens/training/grounding_targets.py` implements the hybrid router:

```python
# For each (sentence, slide) pair:
concept = _detect_concept(sentence)        # → "mask_tumor" | "mask_stroma" | ... | None
if (concept is not None
        and bcss_masks is not None
        and roi_fraction >= MIN_BCSS_COVERAGE_FRACTION   # default 0.2
        and grounding_source == "bcss_hybrid"):
    gt_dist = normalise(bcss_masks[concept])   # BCSS used
else:
    gt_dist = pseudo_gt_cache.get_or_compute(...)  # CONCH pseudo-GT fallback
```

### Known Limitations (Risk 6)

BCSS annotations cover only part of each slide (ROI bounding boxes). The following slides or sentence types always fall back to pseudo-GT:
- Sentences without a recognized concept keyword
- Patches outside the annotated ROI
- Slides without a BCSS annotation file

Additionally, the BCSS cohort may over-represent high-grade / complex-histology cases (TNBC) relative to the full TCGA-BRCA distribution. Monitor per-subtype concept-F1 in evaluation.

---

## CAMELYON16 (Evaluation Only)

CAMELYON16 is used **only for the pointing-game evaluation**. It is not in the training loop.

**Why excluded from training:**
- Binary labels (tumor / non-tumor) — cannot supervise stroma/lymph/necrosis sentences
- Organ mismatch: lymph node metastasis ≠ primary breast tumor histology
- Only ~1 sentence type benefits from binary GT; the rest of the report has no supervision

### Download

```bash
make download-camelyon   # ~150 GB, RunPod only
```

Downloads the test set WSIs and XML polygon annotations.

### XML → Patch-Level Mask Conversion

```bash
uv run python -m patholens.data.camelyon_xml_to_mask \
    --xml-dir data/raw/camelyon16/test/annotations \
    --wsi-dir data/raw/camelyon16/test/images \
    --output-dir data/processed/camelyon_patch_masks \
    --patch-size 448
```

Output per slide:
```
data/processed/camelyon_patch_masks/{slide_id}.h5
├── coordinates: (N_patches, 2)
├── tumor_mask:  (N_patches,) uint8   # 1 = patch overlaps tumor polygon
└── metadata:    {slide_dim, mpp, polygon_count}
```

---

## SlideInstruction

Open dataset from the SlideChat team (`General-Medical-AI/SlideChat` on HuggingFace):
- 4.2K WSI-caption pairs
- 176K VQA examples
- Covers the majority of TCGA-BRCA cases
- GPT-4 structured format

### Download + Filtering

```bash
make prepare-instruction
```

Pipeline (`scripts/04_prepare_slideinstruction.py`):
1. Download from HuggingFace
2. Filter to the 151 BCSS-overlap slide IDs
3. 70/15/15 train/val/test split

Output:
```
data/processed/slideinstruction/train.json   (~105 slides)
data/processed/slideinstruction/val.json     (~23 slides)
data/processed/slideinstruction/test.json    (~23 slides)
```

### Fullscale Training Pool (Published Results)

The capstone headline results come from a **50-slide subset** — the intersection of BCSS-annotated slides, SlideInstruction captions, and successfully downloaded GDC files. This is smaller than the full 151-slide plan due to GDC availability at the time of the run.

| Split | Slides | Config |
|---|---|---|
| Train | 35 | `instruction_train: data/processed/slideinstruction_fullscale/train.json` |
| Val | 8 | `instruction_val: data/processed/slideinstruction_fullscale/val.json` |
| Test | 7 | deterministic, seed=42 |

Config file: `configs/fullscale.yaml`

The `slideinstruction_fullscale/` splits are a filtered subset of `slideinstruction/` containing only the 50 slides with available embeddings. They were generated by running the prepare-instruction script with `--slide-ids` restricted to the available HDF5 files.

> **Note on BCSS key names:** During the fullscale evaluation, a bug was found where the BCSS HDF5 keys are prefixed (`mask_tumor`, `mask_stroma`, ...) but the concept lookup used bare names (`tumor`, `stroma`, ...). This caused the first pointing-game run to report 0/71 concept-bearing sentences. Fixed in `evaluation/bcss_pointing_game.py` (v2 run); the published PG@K numbers are from the corrected v2 run.

### Format

```json
{
  "slide_id": "TCGA-AR-A1AH-01Z-00-DX1",
  "caption": "Invasive ductal carcinoma, Grade 2, with focal lymphovascular invasion...",
  "instruction": "Generate a structured pathology report for this whole-slide image.",
  "vqa_pairs": [
    {"question": "What is the histologic type?", "answer": "Invasive ductal carcinoma"}
  ],
  "metadata": {"histological_type": "IDC", "grade": "G2", "stage": "Stage IIA"}
}
```

---

## Embedding Precompute Pipeline

Single command: `make precompute` (RunPod, ~10 hours for all 151 slides)

```bash
uv run python -m patholens.data.precompute_embeddings --config configs/precompute.yaml
```

### Steps Per Slide

**1. Tissue Segmentation**
```python
mask = otsu_tissue_mask(thumbnail_at_1_25x)
if mask.tissue_fraction < 0.05:
    mask = grandqc_tissue_mask(thumbnail)
```

**2. Patch Extraction**
```python
patches, coords = extract_patches(wsi, mask, patch_size=448, magnification=20)
# Retains patches with tissue_ratio > 0.5
# Typical: 3K–7K patches
```

**3. Dual Patch Embedding**

Two separate encoders are run on the same patches:
```python
# CONCHv1.5 (768-dim) → LLM vision tokens + TITAN input
patch_embeddings_v15 = conchv15_encoder.encode_patches(patches)   # (N, 768)

# CONCH v1 (512-dim) → pseudo-GT cosine similarity
patch_embeddings_v1 = conchv1_encoder.encode_patches(patches)    # (N, 512)
```

> These two embedding spaces are **not compatible** — do not mix them. CONCHv1.5 is vision-only; CONCH v1 has a shared vision+text space used for cross-modal cosine similarity in pseudo-GT computation.

**4. TITAN Slide Encoding**
```python
slide_tokens = titan_encoder.encode_slide(patch_embeddings_v15, coords)  # (M, 768)
# Stored in HDF5 for future use; not used in current training loop
```

**5. HDF5 Output**

```
data/processed/embeddings/tcga_brca/{slide_id}.h5
├── patch_embeddings_v15: (N_patches, 768) float16   ← LLM vision tokens
├── patch_embeddings_v1:  (N_patches, 512) float16   ← pseudo-GT
├── slide_tokens:         (M, 768) float16            ← TITAN (stored, not used in training)
├── coordinates:          (N_patches, 2) int32
├── tissue_mask_lowres:   (H_thumb, W_thumb) uint8
└── attrs: {slide_id, magnification, patch_size, n_patches, n_slide_tokens, model_versions}
```

---

## Data Splits

| Split | TCGA-BRCA (151 BCSS slides) | CAMELYON16 | Purpose |
|-------|----------------------------|------------|---------|
| Train | 105 slides (70%) | — | Caption + grounding training |
| Val | 23 slides (15%) | — | Hyperparameter tuning, early stopping |
| Test in-domain | 23 slides (15%) | — | Caption metrics (BLEU/ROUGE) |
| Test grounding | — | ~100 slides | Pointing game (cross-dataset generalization) |

CAMELYON16 is eval-only — this enables a cross-dataset generalization narrative without data contamination.

---

## Storage Layout

```
data/
├── raw/                              # RunPod only (gigabytes, never synced)
│   ├── tcga_brca/                    # *.svs (151 slides, ~255 GB)
│   ├── camelyon16/
│   │   └── test/
│   │       ├── images/               # *.tif (~100 slides)
│   │       └── annotations/          # *.xml
│   └── bcss/                         # downloaded ROI tiles + label PNGs
│
├── processed/                        # Synced local ↔ RunPod (~35 GB)
│   ├── embeddings/
│   │   ├── tcga_brca/                # {slide_id}.h5 (151 files)
│   │   └── camelyon16/               # {slide_id}.h5 (eval only)
│   ├── bcss_patch_masks/             # {slide_id}.h5 (per-class patch masks)
│   ├── pseudo_gt_cache/              # {slide_id}/{sentence_hash}.npz
│   ├── camelyon_patch_masks/         # {slide_id}.h5 (XML-derived binary GT)
│   └── slideinstruction/
│       ├── train.json
│       ├── val.json
│       └── test.json
│
└── metadata/
    ├── bcss_slide_ids.txt            # 151 TCGA-BRCA slide IDs with BCSS annotations
    ├── tcga_brca_clinical.tsv
    ├── splits.json
    └── case_selection.csv
```

---

## QC Checks

`scripts/qc_embeddings.py` runs after precompute:
- Patch count per slide: 1,000–15,000
- HDF5 file size: 5–30 MB per slide
- No NaN/Inf values in any embedding array
- Coordinates within slide bounds
- BCSS mask N_patches matches embedding N_patches (alignment check)

---

## Privacy / Compliance

- TCGA-BRCA: fully anonymized, public (NIH policy)
- BCSS: CC0 license (public domain)
- CAMELYON16: public, no restrictions
- No PHI is processed
- WSI files remain on RunPod; only embeddings (irreversible) are synced locally

---

## HuggingFace Gated Access

CONCHv1.5 and TITAN are gated models. See `docs/MODEL_ACCESS.md` for access instructions. Ensure `HF_TOKEN` is set in `.env` before any precompute step.
