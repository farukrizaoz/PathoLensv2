# PathoLens-VLM

> Spatially-grounded vision-language model for whole-slide breast cancer histopathology report generation.
> ITÜ AI & Data Engineering capstone, Mayıs-Haziran 2026 (38 gün).
> Faruk Rıza Öz · Emir Arda Eker · Sup: Prof. Dr. Behçet Uğur Töreyin

## Araştırma Sorusu (1 cümle)

WSI'dan üretilen her klinik cümlenin slide'ın hangi bölgesinden geldiğini explicit olarak garanti eden, faithfulness metriği ile ölçülebilen, eğitilebilir bir VLM.

## Çekirdek Katkı

TITAN (MahmoodLab, açık) slide encoder'ı üzerine **grounding-aware report generation head** + **Llama-3.2-3B + LoRA** ekleyerek histopatoloji raporlamada **per-sentence spatial grounding** sağlamak. Mevcut work'lerdeki boşluk: TITAN retrieval/classification için pretrained ama report generation için optimize değil; SlideChat caption üretiyor ama grounding garantisi yok; MedGround radyolojide grounding yapıyor ama histopatolojiye yok.

## Mimari (1-pass özet)

```
WSI (.svs)
  └─ Tissue seg (Otsu / GrandQC fallback)
     └─ Patch extraction (448×448 @20×, ~3K-7K patch/slide)
        └─ CONCHv1.5 patch encoder (frozen)
           └─ TITAN slide encoder (frozen) → slide tokens [N_tokens × 768]
              └─ Linear adapter (TRAIN) → Llama hidden dim [3072]
                 └─ Llama-3.2-3B-Instruct + LoRA r=16 (TRAIN)
                    ├─ Generated structured report
                    ├─ Self-attention extract (text→vision tokens)
                    │  → Per-sentence grounding heatmap
                    └─ → FHIR DiagnosticReport JSON
```

**Eğitilen:** adapter (~5M param) + LoRA (~10M param). Toplam ~15M trainable.
**Frozen:** CONCHv1.5 (304M), TITAN slide encoder, Llama-3.2-3B base.

Detay: `docs/ARCHITECTURE.md`.

## Veri (özet)

| Kaynak | Miktar | Amaç |
|--------|--------|------|
| TCGA-BRCA | 150 slide + reports | Ana fine-tune |
| CAMELYON16 (test) | ~100 slide + pixel mask | Pointing game eval |
| SlideInstruction | 4.2K WSI-caption + 176K VQA | Instruction tuning data (açık, GPT-4 ile yapılandırılmış) |

Detay: `docs/DATA.md`.

## Stack

- Python 3.11+, PyTorch 2.5+, CUDA 12.4 (RunPod), uv paket yöneticisi
- HF: `transformers`, `peft`, `accelerate`, `bitsandbytes` (RunPod-only)
- Modeller: `MahmoodLab/conchv1_5`, `MahmoodLab/TITAN`, `meta-llama/Llama-3.2-3B-Instruct` (hepsi gated, kurumsal HF account)
- WSI: `openslide-python`, `opencv-python-headless`, `h5py`
- MLOps: `wandb`
- Eval: BLEU/ROUGE (HF `evaluate`), custom pointing-game

## Ortam (lokal vs RunPod)

**Lokal (macOS, 20GB AMD GPU + 32GB RAM):**
- Tüm kod yazımı, test, küçük analiz, notebook
- Lokal'de embedding HDF5 inceleme, görselleştirme
- Llama-3.2-3B 4-bit ile inference test (CPU veya küçük örnek)

**Cloud (RunPod, A6000 48GB ~$0.44/sa):**
- Embedding precompute (CONCHv1.5 + TITAN forward pass)
- LoRA fine-tune
- Evaluation runs

**Toplam bütçe:** ~70 GPU saat × $0.44 ≈ **$30-35**.

## Kritik Komutlar (hepsi `make` ile)

```bash
make install              # uv ile bağımlılık + pre-commit
make test                 # smoke tests (no GPU, no data)
make lint && make format  # ruff
make sync-up              # rsync local → RunPod
make sync-down            # checkpoints + logs RunPod → local
make precompute           # CONCHv1.5+TITAN embedding (RunPod)
make train-baseline       # caption-only baseline
make train-grounded       # grounding loss ile
make eval                 # full eval suite
```

## İş Bölümü

**Faruk (Vision):** WSI preprocessing, CONCHv1.5+TITAN embedding pipeline, grounding loss tasarımı, attention extraction, CAMELYON16 evaluation, faithfulness metric.

**Emir (Language):** Llama-3.2 LoRA setup, instruction-following format, SlideInstruction parsing, FHIR DiagnosticReport templating, caption metrics framework.

## Claude Code için Kurallar

**Bunları her session'da takip et:**

1. **Büyük dosyalara dokunma.** `data/`, `checkpoints/`, `wandb/`, `*.h5`, `*.svs`, `*.safetensors` — `.claudeignore` listeli, contexte yükleme.
2. **Secret koruması.** `HF_TOKEN`, `WANDB_API_KEY` `.env`'de. `.env` git-ignored. ASLA commit yok.
3. **Mimari değişikliği yapma** önce bana sormadan: backbone değişimi, LLM seçimi, loss formülasyonu. `docs/ARCHITECTURE.md` source-of-truth.
4. **Cloud-lokal disiplin.** Kod lokal yazılır, `make sync-up` ile RunPod'a gider, eğitim orada çalışır. Lokal'de eğitim başlatma.
5. **Test eklemeden eğitim başlatma.** Yeni loss/dataset/model → `tests/test_smoke.py`'ye minimal smoke test ekle.
6. **Lint disiplin.** Her commit öncesi `make lint`. Ruff hatasıyla commit etme.
7. **Experiment log otomatik.** Her training run sonrası `/log-experiment` slash command'ı çalıştır → `docs/EXPERIMENTS.md` güncellenir.
8. **Token tasarrufu.** Uzun dosyaları `view` ile range belirleyerek oku. Tüm repo'yu okuma. İhtiyacın olan dosyayı bulmak için `Glob` veya `Grep` kullan.

## Slash Skilleri (Token-Optimized Agentler)

`.claude/skills/<name>/SKILL.md` altında tanımlı. Her skill **tek odaklı, dar kapsamlı**, ham komutları sarar:

- `/setup-env` — Lokal veya RunPod env kurulumu (uv, .env, HF login)
- `/precompute-embeddings` — Embedding precompute pipeline (RunPod'da)
- `/train-baseline` — Caption-only baseline başlat
- `/train-grounded` — Grounding loss ile train başlat
- `/eval-pointing-game` — CAMELYON16 pointing game eval
- `/sync-runpod` — local ↔ RunPod sync (push/pull seçimli)
- `/log-experiment` — Son run'ı `EXPERIMENTS.md`'ye işle
- `/diagnose-run` — OOM/CUDA/Loss explosion debug

Her skill kendi kullanım kurallarını içerir. Detaylar için skill'i invoke et, doc gezme.

## Bağımsız Doküman Dosyaları

| Dosya | İçerik |
|-------|--------|
| `docs/ARCHITECTURE.md` | Detaylı mimari kararlar, loss tasarımı, hyperparameter rationale |
| `docs/DATA.md` | TCGA + CAMELYON + SlideInstruction download, preprocessing detayları |
| `docs/ROADMAP.md` | 38 günlük gün-gün plan, sorumlulukları |
| `docs/REMOTE_WORKFLOW.md` | RunPod kurulumu, sync, debug |
| `docs/EXPERIMENTS.md` | Tüm training run sonuçları (auto-updated) |
| `docs/MODEL_ACCESS.md` | HF gated model başvuru talimatları |

## Faydalı Linkler

- TITAN: https://huggingface.co/MahmoodLab/TITAN
- CONCHv1.5: https://huggingface.co/MahmoodLab/conchv1_5
- Llama-3.2-3B: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- TITAN paper: https://arxiv.org/abs/2411.19666
- SlideChat (referans, kullanmıyoruz): https://arxiv.org/abs/2410.11761
- MedGround (grounding ref): https://arxiv.org/abs/2601.06847
- Direct Visual Grounding (KL Attention Loss, 2025): https://arxiv.org/abs/2511.12738
