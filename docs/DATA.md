# Data

Veri kaynakları, indirme talimatları, preprocessing, storage layout.

## Genel Bakış

| Dataset | Kullanım | Raw boyut | Processed | Lokasyon |
|---------|----------|-----------|-----------|----------|
| TCGA-BRCA (150 slide) | Ana fine-tune | ~250 GB | ~20 GB embedding HDF5 | RunPod (raw), local (processed) |
| CAMELYON16 (test, ~100 slide) | Pointing game eval | ~150 GB | ~10 GB embedding HDF5 | RunPod (raw), local (processed) |
| SlideInstruction | Instruction tuning | ~5 GB JSON | ~5 GB | Local + RunPod |

**Disiplin:** Ham WSI dosyaları **sadece RunPod'da**. Local'e sadece embedding HDF5 + metadata sync edilir. Bu sayede local makine ~30 GB ile kurtarır.

---

## TCGA-BRCA

### Slide Selection

**Stratifiye seçim** (raporlama çeşitliliği için):
- IDC (Invasive Ductal Carcinoma): 90 slide (~%60)
- ILC (Invasive Lobular Carcinoma): 30 slide (~%20)
- Diğer (mucinous, medullary, mixed): 30 slide (~%20)

Grade dağılımı hedef: G1=30, G2=60, G3=60.

### Indirme

```bash
# RunPod'da çalışır, ~12-24 saat
make download-tcga
```

Adımlar (otomatize, `scripts/01_download_tcga.sh`):

1. GDC manifest oluştur (klinik filtreleme + 150 slide stratifiye seçim)
2. `gdc-client` ile paralel indirme (8 worker)
3. Klinik metadata TSV indirme (case_id, grade, subtype, etc.)

```bash
# Manifest + clinical metadata location
data/raw/gdc_manifest_BRCA_150.txt
data/raw/tcga_brca_clinical.tsv
```

### Klinik Etiketler

GDC clinical TSV'den önemli alanlar:
- `case_id` (UUID, slide ID ile eşleşir)
- `histological_type`
- `tumor_grade`
- `pathologic_stage`
- `er_status`, `pr_status`, `her2_status`

### Pathology Reports

**Karar: TCGA pathology raporlarını DOĞRUDAN parse etmiyoruz.**
SlideInstruction zaten TCGA-BRCA için GPT-4 ile yapılandırılmış caption'lar sunuyor (4.2K caption içinde TCGA-BRCA case'leri büyük çoğunluğu kapsıyor). Bu sorunu çözüyor.

Eğer ekstra rapor verisi gerekirse Hafta 4'te GDC API ile PDF reports indirilebilir, ama şu an scope dışı.

---

## CAMELYON16

### Indirme

```bash
make download-camelyon  # ~150 GB
```

İçerik (`scripts/02_download_camelyon.sh`):
- WSI dosyaları (TIFF format)
- Pixel-level tumor annotation (XML format)
- Slide-level binary label

**Sadece test set (~130 slide) indirilir, training set indirilmez.**

Kaynak: https://camelyon16.grand-challenge.org/Data/
AWS S3 mirror: `s3://camelyon-dataset/CAMELYON16/`

### XML → Patch-level Mask Conversion

CAMELYON16 XML annotation'ları polygon olarak tumor regions tanımlar. Grounding ground truth için patch-level binary mask gerekli:

```bash
uv run python -m patholens.data.camelyon_xml_to_mask \
    --xml-dir data/raw/camelyon16/test/annotations \
    --wsi-dir data/raw/camelyon16/test/images \
    --output-dir data/processed/camelyon_patch_masks \
    --patch-size 448
```

Output (HDF5, slide başına):
```
test_001.h5
├── coordinates: (N_patches, 2)  # patch (x, y) origin
├── tumor_mask: (N_patches,) uint8  # 1 if patch overlaps tumor polygon
└── metadata: {slide_dim, mpp, polygon_count}
```

---

## SlideInstruction

### Description

SlideChat takımının açık datasetı:
- 4.2K WSI-caption pair
- 176K VQA örneği
- TCGA-BRCA case'lerini büyük çoğunluğu kapsıyor
- GPT-4 ile yapılandırılmış format

Source: `General-Medical-AI/SlideChat` HuggingFace dataset repo.

### Indirme + Filtreleme

```bash
make prepare-instruction
```

Pipeline (`scripts/04_prepare_slideinstruction.py`):
1. HuggingFace'ten download
2. TCGA-BRCA case ID'lerine göre filtrele (sizin 150 slide ile eşleşen)
3. Train/val/test split oluştur (80/10/10)
4. Output: `data/processed/slideinstruction/{train,val,test}.json`

### Format

```json
{
  "slide_id": "TCGA-AR-A1AH-01Z-00-DX1",
  "case_id": "TCGA-AR-A1AH",
  "caption": "Invasive ductal carcinoma, Grade 2, with focal lymphovascular invasion. Tumor measures 2.3 cm. Margins negative. No metastatic carcinoma in 12 lymph nodes.",
  "vqa_pairs": [
    {"question": "What is the histologic type?", "answer": "Invasive ductal carcinoma"},
    {"question": "Nottingham grade?", "answer": "Grade 2"}
  ],
  "metadata": {
    "histological_type": "IDC",
    "grade": "G2",
    "stage": "Stage IIA"
  }
}
```

---

## Preprocessing Pipeline

Tek komut: `make precompute`

```bash
# RunPod'da çalışır, tüm slide'lar için ~10 saat
uv run python -m patholens.data.precompute_embeddings \
    --config configs/precompute.yaml
```

Pipeline adımları (slide başına):

### 1. Tissue Segmentation

```python
# Otsu first
mask = otsu_tissue_mask(wsi_thumbnail @ 1.25x)
if mask.sum() < min_tissue_area:
    mask = grandqc_tissue_mask(wsi_thumbnail)  # fallback
```

### 2. Patch Extraction

```python
# Tissue mask üzerinde 448×448 patch grid @20×
# Tissue ratio > 0.5 olan patch'ler tutulur
patches, coords = extract_patches(wsi, mask, patch_size=448, mag=20)
# Tipik: ~3K-7K patch
```

### 3. CONCHv1.5 Patch Embedding

```python
# Batch processing, mixed precision (bf16)
patch_embeddings = []
for batch in batched(patches, batch_size=128):
    emb = conchv15_encoder(batch)  # (B, 768)
    patch_embeddings.append(emb.float())
# Total: (N_patches, 768)
```

### 4. TITAN Slide Encoding

```python
# TITAN slide encoder positional encoding ile slide-level tokens üretir
slide_tokens = titan_slide_encoder(patch_embeddings, coords)
# Output: (N_slide_tokens, 768)
```

### 5. HDF5 Cache

Slide başına bir HDF5 dosyası:

```
TCGA-AR-A1AH-01Z-00-DX1.h5
├── patch_embeddings: (N_patches, 768) float16
├── slide_tokens: (N_slide_tokens, 768) float16
├── coordinates: (N_patches, 2) int32
├── tissue_mask_lowres: (H, W) uint8
└── attrs:
    ├── slide_id, case_id
    ├── magnification: 20
    ├── patch_size: 448
    ├── total_tissue_patches
    ├── n_slide_tokens
    └── conch_version, titan_version
```

---

## Storage Layout

```
data/
├── raw/                          # ONLY on RunPod (gigabytes)
│   ├── tcga_brca/                # *.svs files
│   ├── camelyon16/
│   │   ├── test/
│   │   │   ├── images/           # *.tif
│   │   │   └── annotations/      # *.xml
│   └── gdc_manifest_BRCA_150.txt
│
├── processed/                    # Synced local ↔ RunPod (~30 GB)
│   ├── embeddings/
│   │   ├── tcga_brca/            # *.h5 (one per slide)
│   │   └── camelyon16/           # *.h5
│   ├── tissue_masks/             # numpy arrays (low-res)
│   ├── camelyon_patch_masks/     # *.h5 (XML-derived ground truth)
│   └── slideinstruction/
│       ├── train.json
│       ├── val.json
│       └── test.json
│
└── metadata/
    ├── tcga_brca_clinical.tsv
    ├── splits.json
    └── case_selection.csv
```

---

## Data Splits

| Split | TCGA-BRCA | CAMELYON16 | Amaç |
|-------|-----------|------------|------|
| Train | 105 slide (70%) | — | Caption + grounding loss eğitimi |
| Val | 22 slide (15%) | — | Hyperparameter tuning, early stopping |
| Test (in-domain) | 23 slide (15%) | — | Final caption metrics |
| Test (grounding) | — | ~100 slide | Pointing game ground truth |

CAMELYON16 sadece eval için kullanılır → cross-dataset generalization hikayesi.

**Grounding v2 deneyinde** (Hafta 5) opsiyonel olarak CAMELYON16 train set'ten 30-50 slide eğitime dahil edilebilir → ablation comparison için.

---

## QC Checks

`scripts/qc_embeddings.py` her precompute sonrası otomatik kontrol:
- ✓ Tissue patch sayısı slide başına 1000-15000 arasında
- ✓ Embedding HDF5 boyutu beklenen aralıkta (slide başına 5-30 MB)
- ✓ NaN/Inf yok
- ✓ Coordinate'lar slide boyutu içinde
- ✓ slide_tokens shape (N, 768)

---

## Privacy / Compliance

- TCGA-BRCA tamamen anonim, public, KVKK/GDPR sorunu yok
- CAMELYON16 public, sınırlama yok
- Hiçbir PHI işlenmez
- WSI dosyaları RunPod'da kalır, lokal'e taşınmaz
- Embedding'lerden orijinal WSI rekonstrüksiyonu mümkün değil (irreversible)

---

## HuggingFace Gated Access

CONCHv1.5 ve TITAN modellerinin gated olduğunu unutma. Detay: `docs/MODEL_ACCESS.md`.

İndirme öncesi `huggingface-cli login` çalıştırılmalı veya `HF_TOKEN` env'de set edilmiş olmalı.
