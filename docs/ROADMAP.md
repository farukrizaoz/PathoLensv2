# Roadmap

**Window:** Today  - June 12, 2026
**Buffer:** 0 days. Tight schedule, parallel work essential.

---

## Hafta 1 (Gün 1-7): Setup + Data + HF Access

### Gün 1: HF gated access başvuruları (HEMEN)

**KRİTİK — bu işlem gecikirse tüm pipeline gecikir.**

- [ ] HF account: primary email = `@itu.edu.tr`
- [ ] CONCHv1.5 başvurusu: https://huggingface.co/MahmoodLab/conchv1_5
- [ ] TITAN başvurusu: https://huggingface.co/MahmoodLab/TITAN
- [ ] Llama-3.2-3B agreement: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- [ ] Form'da affiliation: "Istanbul Technical University, Department of AI and Data Engineering"
- [ ] Research use: "Capstone project on grounded report generation for breast cancer histopathology, supervised by Prof. Dr. Behçet Uğur Töreyin"

Detay: `docs/MODEL_ACCESS.md`.

### Gün 1-2: Repo, env, RunPod kurulumu

**İkisi birlikte:**

- [ ] Repo init, GitHub'a push
- [ ] `make install-dev` lokal'de
- [ ] `.env` doldur (HF_TOKEN test için, WANDB_API_KEY)
- [ ] `make test` smoke geçiyor

**RunPod setup:**

- [ ] RunPod hesap, $50 kredi
- [ ] A6000 48GB pod oluştur, 200GB persistent storage
- [ ] SSH key kur, `make ssh` test
- [ ] `bash scripts/00_runpod_setup.sh` (CUDA verify, openslide system lib, uv install)
- [ ] `make sync-up` ile kod push, RunPod'da `make install-gpu`
- [ ] `make gpu-info` doğrula

### Gün 3-5: TCGA-BRCA + CAMELYON16 indirme paralel

**Faruk:**

- [ ] GDC manifest oluştur (150 BRCA slide stratifiye)
- [ ] `make download-tcga` (RunPod, ~12-24 saat, background)
- [ ] CAMELYON16 download script test et (`make download-camelyon`, paralel)

**Emir:**

- [ ] `make prepare-instruction` (SlideInstruction download + filter)
- [ ] Train/val/test split oluştur
- [ ] Caption length distribution notebook

### Gün 6-7: Embedding precompute + Tissue seg

- [ ] HF access onayları geldi mi? (Gün 4-5 civarı bekleniyor)
- [ ] Tissue segmentation pipeline test (5 slide, görsel kontrol)
- [ ] **`make precompute`** çalıştır (RunPod, ~10 saat)
- [ ] HDF5 QC pass
- [ ] `make sync-down` ile embedding'leri lokal'e çek (~25 GB)
- [ ] `notebooks/00_data_exploration.ipynb` ile inceleme

**Hafta 1 Checkpoint:**

- ✓ HF gated access onaylandı
- ✓ 150 TCGA-BRCA + 100 CAMELYON16 indirildi
- ✓ Tüm slide'lar için embedding HDF5 oluştu
- ✓ Lokal-RunPod sync akışı çalışıyor

---

## Hafta 2 (Gün 8-14): Model entegrasyonu + İlk forward pass

### Gün 8-10: TITAN slide encoder + adapter

**Faruk:**

- [ ] `src/patholens/models/titan_encoder.py` — TITAN HF wrapper, slide_tokens API
- [ ] `src/patholens/models/adapter.py` — linear projection (768 → 3072)
- [ ] Smoke test: HDF5'ten patch_emb yükle → TITAN forward → adapter forward, shape correctness

### Gün 11-12: Llama-3.2-3B + LoRA setup

**Emir:**

- [ ] `src/patholens/models/llm_backbone.py` — Llama-3.2-3B + 4-bit quant + LoRA
- [ ] LLaVA-style multimodal input format: `<bos><vision_tokens><instruction>`
- [ ] `src/patholens/models/grounded_vlm.py` — uçtan uca model

### Gün 13-14: İlk forward pass + attention extraction

- [ ] Bir slide → embedding → adapter → Llama → caption üretiyor mu?
- [ ] `output_attentions=True` ile attention extraction çalışıyor mu?
- [ ] `notebooks/01_test_inference.ipynb` ile görsel doğrulama
- [ ] **Karar noktası:** Hangi layer/head grounding-faithful? (5-10 örnekte attention map'leri incele)

**Hafta 2 Checkpoint:**

- ✓ TITAN + adapter + Llama uçtan uca çalışıyor
- ✓ Caption (eğitimsiz, instruction-following base) üretiliyor
- ✓ Attention extraction kodu hazır

---

## Hafta 3 (Gün 15-21): Caption baseline + Grounding extractor

### Gün 15-17: Caption-only baseline eğitimi

- [ ] `configs/caption_baseline.yaml` finalize
- [ ] `src/patholens/training/trainer.py` — HF Trainer wrapper veya custom loop
- [ ] `src/patholens/data/dataset.py` — SlideInstruction → torch Dataset
- [ ] WandB integration test (loss curve, generation samples)
- [ ] **`make train-baseline`** RunPod'da (~12 saat)
- [ ] BLEU/ROUGE baseline skorları kayıtlı

### Gün 18-19: Grounding extractor

**Faruk:**

- [ ] `src/patholens/models/grounding_extractor.py`
- [ ] Sentence boundary detection (spaCy)
- [ ] Per-sentence text→vision attention pooling
- [ ] Vision token → patch coordinate mapping (TITAN attention ile geri çarpım)
- [ ] Görselleştirme: bir cümle + heatmap overlay (notebook)

### Gün 20-21: Pseudo-grounding ground truth

**Faruk + Emir:**

- [ ] CONCHv1.5 text encoder ile sentence embeddings
- [ ] Patch embeddings ile cosine similarity → softmax pseudo-attention
- [ ] Threshold tuning notebook (0.3 vs 0.5)
- [ ] `src/patholens/data/pseudo_grounding.py`

**Hafta 3 Checkpoint:**

- ✓ Caption baseline trained, BLEU/ROUGE kaydedildi
- ✓ Grounding extractor: cümle → heatmap görselleştirme çalışıyor
- ✓ Pseudo-grounding ground truth pipeline hazır

---

## Hafta 4 (Gün 22-28): Grounding loss v1 + CAMELYON eval

### Gün 22-24: Grounding loss v1 implementasyonu

**Faruk:**

- [ ] `src/patholens/training/losses.py` → `GroundingLoss` class (KL divergence)
- [ ] Faithfulness regularization (entropy penalty)
- [ ] `configs/grounding_v1.yaml`
- [ ] Smoke test: loss değerleri makul aralıkta

### Gün 25-26: Grounding eğitimi

- [ ] **`make train-grounded`** RunPod'da (~16 saat)
- [ ] Caption loss vs grounding loss balansı izle (lambda_grounding sweep: 0.1, 0.3, 0.5)
- [ ] Caption kalitesi kötüleşmedi mi (baseline ile compare)?

### Gün 27-28: CAMELYON16 pointing game

**Faruk:**

- [ ] `src/patholens/evaluation/pointing_game.py`
- [ ] **`make eval-pointing`** çalıştır
- [ ] Sonuç: tumor cümleleri için pointing game accuracy (PG@5, PG@10)
- [ ] Baseline vs grounded model karşılaştırma tablosu

**Hafta 4 Checkpoint:**

- ✓ Grounding loss v1 ile eğitilmiş model
- ✓ CAMELYON16 pointing game accuracy ölçüldü
- ✓ Caption metrics degradation kontrolü

---

## Hafta 5 (Gün 29-35): Ablation + Grounding v2 + FHIR

### Gün 29-30: Faithfulness intervention test

**Faruk:**

- [ ] `src/patholens/evaluation/intervention_test.py`
- [ ] Top-K attention patch'leri masking
- [ ] Generation değişikliği ölç (BLEU drop, sentence change rate)
- [ ] Baseline vs grounded faithfulness karşılaştırma

### Gün 31-32: Grounding loss v2

- [ ] `configs/grounding_v2.yaml`: CAMELYON pixel mask explicit supervision
- [ ] CAMELYON train slide'larından 30-50 tane eğitime dahil
- [ ] **`make train-grounded-v2`** RunPod'da (~16 saat)
- [ ] v1 vs v2 karşılaştırma

### Gün 33: FHIR DiagnosticReport templating

**Emir (4-6 saat işi):**

- [ ] `src/patholens/reporting/fhir.py`
- [ ] `fhir.resources.DiagnosticReport` template
- [ ] Generated text → JSON serialization
- [ ] Demo: bir slide için FHIR JSON output
- [ ] Test: `tests/test_fhir.py`

### Gün 34-35: Ablation tablosu + son ayarlamalar

- [ ] Final ablation tablosu:

| Config                         | BLEU-4 | ROUGE-L | PG@5 | Faithfulness | Caption Δ |
| ------------------------------ | ------ | ------- | ---- | ------------ | ---------- |
| Caption baseline               | ...    | ...     | ...  | ...          | 0          |
| + L_grounding (CONCH pseudo)   | ...    | ...     | ...  | ...          | ...        |
| + L_grounding (CAMELYON expl.) | ...    | ...     | ...  | ...          | ...        |
| + L_faithfulness reg.          | ...    | ...     | ...  | ...          | ...        |

**Hafta 5 Checkpoint:**

- ✓ İki grounding loss varyantı, ablation tablosu hazır
- ✓ Pointing game + intervention test sonuçları
- ✓ FHIR JSON output çalışıyor

---

## Hafta 5.5-6 (Gün 36-38): Demo + Rapor + Sunum

### Gün 36: Demo notebook

- [ ] `notebooks/02_grounding_visualization.ipynb`
- [ ] Bir TCGA slide üzerinde:
  - Slide thumbnail
  - Generated structured report
  - Her cümle için heatmap overlay
  - FHIR JSON gösterimi
  - Karşılaştırma: baseline vs grounded
- [ ] CAMELYON için: pixel mask ground truth + heatmap overlay

### Gün 37: Rapor

Yapı:

1. Introduction (problem, gap)
2. Related Work (TITAN, SlideChat, MedGround, KL Attention Loss)
3. Method (TITAN encoder + adapter + Llama LoRA, grounding loss tasarımı)
4. Datasets (TCGA-BRCA, CAMELYON16, SlideInstruction)
5. Experiments (ablation, hyperparameter)
6. Results (caption metrics, pointing game, intervention)
7. Discussion (limitations, future work)
8. Conclusion

Şekiller: mimari diagram, loss curves, ablation tablo, demo screenshots.

### Gün 38: Sunum + final check

- [ ] 15-20 dakikalık sunum slaytları
- [ ] Demo prep: lokal'de inference (gerekirse video kayıt)
- [ ] GitHub repo final temizlik (README, model card, MODEL_ACCESS.md)
- [ ] Profesöre sunum

---

## Risk Planı

| Risk                                          | Olasılık   | Plan B                                                                         |
| --------------------------------------------- | ------------ | ------------------------------------------------------------------------------ |
| HF access geç onay (CONCHv1.5/TITAN)         | Orta         | UNI ile başla, gelince geç. Ek 2-3 gün kayıp.                              |
| TITAN API beklenmedik davranıyor             | Düşük     | TITAN model_card'daki demo notebook ile başla, kendi pipeline'ına entegre et |
| Grounding loss v1 hiç iyileşme yok          | Orta         | KL → MSE → cosine alignment dene. v2'ye paralel git.                         |
| Llama attention extraction zor                | Düşük     | LLaVA literatüründen layer/head referansı var, deneysel seçilir            |
| RunPod kapasite sorunu                        | Düşük     | Lambda Labs A6000 fallback ($0.80/h, biraz pahalı)                            |
| 38 gün yetmiyor (hafta 4 sonu kaçırıldı) | Orta-Yüksek | Grounding v2 atlanır. Minimum: baseline + grounding v1 + pointing game.       |

## Minimum Defansif Final Ürün

Eğer her şey kötüye giderse, sunulabilir minimum:

1. Caption-only baseline trained
2. Grounding loss v1 trained
3. CAMELYON16 pointing game accuracy karşılaştırması
4. 5-10 slide üzerinde demo (heatmap görselleştirme)
5. FHIR output örneği

Bu bile bir bitirme projesi olarak tamamen savunulabilir.
