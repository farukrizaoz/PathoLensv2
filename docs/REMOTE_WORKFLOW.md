# Remote Workflow (RunPod)

Lokal kod yazma + RunPod GPU eğitimi akışı.

## Kuruluş Felsefesi

```
┌──────────────────┐         ┌─────────────────────┐
│   LOKAL          │         │   RUNPOD            │
│   (macOS)        │         │   (A6000 48GB)      │
├──────────────────┤         ├─────────────────────┤
│                  │         │                     │
│ • Kod yazma      │ rsync   │ • Embedding precom. │
│ • Testler        │ ────►   │ • LoRA fine-tune    │
│ • Notebook       │         │ • Evaluation        │
│ • Görselleşt.    │  rsync  │ • Heavy GPU work    │
│ • Embedding      │ ◄────   │                     │
│   inceleme       │         │                     │
└──────────────────┘         └─────────────────────┘
```

**Kural:** Eğitim ve veri işleme **sadece RunPod**. Kod yazma ve hızlı iter **sadece lokal**.

---

## RunPod Pod Oluşturma

### Önerilen Konfigürasyon

| Setting | Value |
|---------|-------|
| GPU | RTX A6000 48GB (~$0.44/h) veya RTX 4090 24GB (~$0.34/h, daha sıkı) |
| Container Disk | 50 GB (sadece env için) |
| Volume Disk (persistent) | **200 GB** (data + checkpoints) |
| Image | `runpod/pytorch:2.5.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Region | EU veya US-East (Türkiye'ye yakın) |

**Volume önemli:** Pod kapatıldığında container disk silinir, volume kalır. Tüm `data/` ve `checkpoints/` `/workspace/` (volume mount) altında olmalı.

### SSH Setup

1. RunPod Console → Pod → "Connect" → "SSH over exposed TCP"
2. Komutu kopyala: `ssh root@xxx.proxy.runpod.net -p NNNNN -i ~/.ssh/id_rsa`
3. `.env`'e ekle:
   ```bash
   RUNPOD_HOST=root@xxx.proxy.runpod.net
   RUNPOD_PORT=NNNNN
   ```
4. Test: `make ssh`

### İlk Pod Setup

İlk SSH'tan sonra:

```bash
cd /workspace
git clone <your-repo-url> patholens-vlm
cd patholens-vlm
bash scripts/00_runpod_setup.sh   # CUDA verify, openslide, uv install
make install-gpu                   # GPU extras dahil
make gpu-info                      # Sanity check
```

`scripts/00_runpod_setup.sh` şunları yapar:
- `apt-get install openslide-tools` (sistem kütüphanesi)
- `curl -fsSL https://astral.sh/uv/install.sh | sh` (uv install)
- HF login (token .env'den)
- WandB login

---

## Günlük Akış

```bash
# === Lokal'de ===
# Kod değişikliği yap, test et
make test
make lint

# RunPod'a push et
make sync-up

# === RunPod'da (SSH) ===
ssh -p $RUNPOD_PORT $RUNPOD_HOST
cd /workspace/patholens-vlm

# Eğitim başlat (tmux içinde, pod kapanırsa devam etsin diye)
tmux new -s train
make train-grounded
# Ctrl+B, D ile detach

# === Lokal'de ===
# Sonuçları çek
make sync-down

# Lokal'de notebook'larda analiz
make notebook
```

---

## tmux Disiplin (kritik)

**Eğitim ASLA tmux dışında çalıştırılmamalı.** RunPod web SSH timeout yapabilir veya pod kapatılabilir, eğitim ortada kalır.

```bash
# Yeni session
tmux new -s train

# Detach (eğitim devam eder)
Ctrl+B, D

# Re-attach
tmux attach -t train

# Tüm session'ları listele
tmux ls

# Session öldür
tmux kill-session -t train
```

---

## Storage Yönetimi (200 GB volume)

| İçerik | Boyut | Lokasyon |
|--------|-------|----------|
| TCGA-BRCA WSI (150 slide) | ~250 GB | **PROBLEM** |
| CAMELYON16 test | ~150 GB | **PROBLEM** |
| Embedding HDF5 | ~30 GB | OK |
| Checkpoints | ~5 GB | OK |
| Logs | ~1 GB | OK |

**Toplam ~440 GB > 200 GB volume.**

**Çözüm: WSI'lar streaming download + on-the-fly processing**
1. Slide'ları batch halinde indir (10 slide grupları)
2. Her batch için tissue seg + patch + embedding precompute
3. Embedding HDF5 kaydet, ham WSI sil
4. Bir sonraki batch'e geç

`scripts/01_download_tcga.sh` bu mantığı uygular (`--streaming` flag).

CAMELYON16 için aynı: 10 slide'lık batch, embedding sonrası WSI sil.

Bu sayede peak disk kullanımı: ~50 GB (10 slide WSI + embedding biriktirme).

---

## Debugging RunPod'da

### OOM (Out of Memory)

```bash
# VRAM monitör
watch -n 1 nvidia-smi

# Eğitim sırasında oom olursa:
# 1. batch_size düşür (zaten 1)
# 2. gradient_accumulation artır
# 3. gradient_checkpointing aktif et (zaten on)
# 4. mixed precision bf16 → fp16 (bazen daha az VRAM)
```

`/diagnose-run` slash command'ı bu adımları rehberli şekilde işler.

### Sıkışan tmux session

```bash
tmux kill-server  # tüm tmux temiz
```

### Disk doldu

```bash
df -h /workspace
# Embedding precompute sırasında ham WSI silindiği doğrula
ls -lah data/raw/tcga_brca | head
# Eski checkpoint sil (en yeni 2 tut)
ls -t checkpoints/ | tail -n +3 | xargs -I {} rm -rf checkpoints/{}
```

---

## Pod Kapatma Disiplini

**Para tasarrufu için:**

```bash
# Eğitim bittikten sonra ÖNCE sync-down yap (lokal'de)
make sync-down

# Sonra pod'u STOP et (DELETE değil — volume kalır)
# RunPod console → Pod → Stop
```

**Stop:** Compute durur, sadece volume için ödersin (~$0.05/GB/ay → 200 GB = $10/ay).
**Delete:** Her şey gider, geri alınamaz.

Sadece tüm proje bittiğinde delete.

---

## Maliyet Takibi

```bash
# RunPod console → Billing → Invoices
# veya
runpodctl get pod   # CLI ile mevcut maliyet
```

Hedef: toplam $35-40 (eğitim) + $10 (storage). Total <$50.

---

## Bütün Akış Özet

```
HAFTA 1:
  Pod oluştur → setup → data download → embedding precompute → sync-down
HAFTA 2-5:
  Lokal kod yaz → sync-up → tmux'ta eğit → sync-down → analiz
HAFTA 6:
  Final sync-down → pod STOP (silme!)
PROJE BİTTİ:
  Pod DELETE
```
