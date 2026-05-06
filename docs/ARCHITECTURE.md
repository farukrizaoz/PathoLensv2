# Architecture

Detaylı mimari kararları, design rationale, hyperparameter seçimleri.

## Tam Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  WSI input (.svs, ~50K × 50K pixels @20×)                            │
│       │                                                              │
│       ▼                                                              │
│  Tissue Segmentation (Otsu @ 1.25× / GrandQC fallback)               │
│       │                                                              │
│       ▼                                                              │
│  Patch extraction: 448×448 RGB tiles @20× from tissue regions        │
│       │ (~3K-7K patches per slide)                                   │
│       ▼                                                              │
│  CONCHv1.5 ViT-L patch encoder  [FROZEN]                             │
│       │ patch_embeddings: (N_patches, 768)                           │
│       ▼                                                              │
│  TITAN Slide Encoder  [FROZEN]                                       │
│       │ Hierarchical aggregation with positional encoding            │
│       │ slide_tokens: (N_slide_tokens, 768)                          │
│       │ N_slide_tokens depends on patch grid (typically 256-1024)    │
│       ▼                                                              │
│  Linear Adapter  [TRAINABLE, ~5M params]                             │
│       │ Projection: 768 → 3072 (Llama hidden dim)                    │
│       │ adapter_tokens: (N_slide_tokens, 3072)                       │
│       ▼                                                              │
│  Llama-3.2-3B-Instruct + LoRA (r=16) [TRAINABLE LoRA, ~10M params]   │
│       │                                                              │
│       │ Input format (LLaVA-style):                                  │
│       │   <bos><adapter_tokens><instruction_text>                    │
│       │                                                              │
│       │ Vision tokens placed in same sequence as text tokens.        │
│       │ Causal self-attention attends across both.                   │
│       │                                                              │
│       │ Output: generated tokens (autoregressive)                    │
│       │                                                              │
│       ├───▶ Generated structured report                              │
│       │                                                              │
│       └───▶ Self-attention extraction                                │
│              │ For each generated token, extract attention weights   │
│              │ over input vision tokens.                             │
│              │ Aggregate by sentence boundaries (spaCy).             │
│              │                                                       │
│              ▼                                                       │
│           Per-sentence grounding heatmap                             │
│              │ Map back to original patches via TITAN attention      │
│              ▼                                                       │
│           Spatial overlay on WSI                                     │
│                                                                      │
│  ┌───────────────────────────────────────┐                           │
│  │ FHIR DiagnosticReport JSON serializer │ ← reporting/fhir.py       │
│  └───────────────────────────────────────┘                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Komponent Kararları

### Tissue Segmentation: Otsu + GrandQC (fallback)

**Otsu (default):** Hızlı, 1.25× thumbnail üzerinde HSV saturation thresholding. Çoğu temiz TCGA slide için yeterli.

**GrandQC (fallback):** Eğer Otsu artifact'lara duyarlı çıkıyorsa (ink, bubble, fold) GrandQC pretrained tissue segmentation kullanılır. GrandQC HuggingFace üzerinden public.

**Karar mantığı:** Hafta 1'de 5-10 slide'da görsel kontrol et. Otsu mask kabul edilebilir görünüyorsa devam, değilse GrandQC.

### Patch boyutu: 448×448 @20×

**Neden 448 değil 256:**
- CONCHv1.5'in eğitildiği boyut **448×448** (CONCH v1 ise 224×224)
- TITAN patch features 448×448 üzerine inşa edilmiş, başka boyut kullanırsan TITAN çalışmaz
- Stride = 448 (no overlap), embedding sayısını minimize eder

Tipik bir TCGA-BRCA slide'da: ~3K-7K patch (256×256 olsaydı 5K-10K olurdu).

### Patch Encoder: CONCHv1.5 (ViT-L)

CONCHv1.5 tercih nedeni:
- TITAN ile **uyumluluk zorunluluğu** (TITAN sadece v1.5 patch features ile çalışır)
- ViT-L (304M param) > ViT-B (CONCH v1, 86M)
- Embedding dim: 768 (CLAM ve standart MIL boruları ile uyumlu)
- Multimodal pretraining (vision + text) → text encoder grounding pseudo-supervision için kullanılabilir

**Frozen.** Embedding precompute → HDF5 cache → eğitim hızlı.

**Lisans:** CC-BY-NC-ND 4.0, research-only, OK.

### Slide Encoder: TITAN (frozen)

TITAN seçimi rationale:
- **Slide-level pretrained transformer** — 335K WSI üzerinde self-supervised + vision-language alignment
- ABMIL veya kendi attention aggregator yazma ihtiyacını ortadan kaldırır
- 2D positional encoding ile patch grid'in spatial yapısını korur (önemli: grounding için)
- Slide tokens output: text-aligned (TITAN'ın kendi text encoder'ı PathChat caption'larıyla eğitilmiş)

**Alternatif düşünüldü, reddedildi:**
- Sıfırdan ABMIL: TITAN'ın 335K slide pretraining'inden gelen kazancı atmak demek
- LongNet (SlideChat'ten): SlideChat checkpoint açık değil
- ACMIL/TransMIL: 38 günde sıfırdan eğitmek riski yüksek

**Frozen.** TITAN slide encoding bir kez yapılır, output cache'lenir.

### Adapter: Linear Projection (LLaVA-style)

```python
class Adapter(nn.Module):
    def __init__(self, in_dim=768, out_dim=3072):
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
```

**Q-Former yerine linear neden:**
- LLaVA-1.5 paper'ı linear ≈ Q-Former performans gösterdi
- 5M parametre, eğitimi hızlı
- Mimari sadeliği = debug kolaylığı
- 38 günlük zamana uygun

Eğer Hafta 5'te zaman kalırsa Q-Former ablation eklenebilir, ama mecburi değil.

### LLM: Llama-3.2-3B-Instruct

| Spec | Value |
|------|-------|
| Parameter | 3.21B |
| Hidden dim | 3072 |
| Layers | 28 |
| Heads | 24 |
| Context | 128K (kullanmıyoruz, ~2K yeter) |
| License | Meta Llama 3.2 Community |

**LoRA config:**
```yaml
r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
bias: "none"
task_type: "CAUSAL_LM"
```

Trainable LoRA params: ~10M. Toplam adapter+LoRA: ~15M.

**Neden 7B değil 3B:**
- 3B + 4-bit + LoRA → 6-8 GB VRAM, A6000 48GB'da bol
- Eğitim hızı 7B'nin 2x'i (önemli: 38 gün)
- Tıbbi domain'de 3B vs 7B farkı LoRA fine-tune ile büyük ölçüde kapanıyor
- Lokal'de bile (20GB AMD GPU, ROCm) inference test edilebilir

**Neden BioMistral / Med-Llama yok:**
- 2024 başı modeller, eski
- Llama-3.2-3B base modern, ondan medical fine-tune yapmak daha sağlıklı
- Bizim domain WSI captioning, genel medical QA değil

## Loss Tasarımı

```
L_total = L_caption + λ_g · L_grounding + λ_f · L_faithfulness
```

Default: `λ_g = 0.3, λ_f = 0.1`. Ablation Hafta 5'te.

### L_caption — Standard LM Cross-Entropy

```python
# Padding ve image token positions mask edilir
loss = F.cross_entropy(logits[mask], targets[mask], ignore_index=-100)
```

Ground truth: SlideInstruction caption'ları (TCGA-BRCA filtered).

### L_grounding — Sentence-Patch Alignment

**Konsept:** Her üretilen cümle için, modelin vision token'lara verdiği attention dağılımı, beklenen "ground truth attention" ile hizalansın.

**Ground truth attention iki kaynaktan:**

**(a) CONCHv1.5 zero-shot pseudo-supervision (TCGA için):**
```python
# Her cümle için CONCH text embedding
sentence_emb = conch_text_encoder(sentence)  # (768,)
# Patch'lerle cosine similarity
patch_emb = conch_patch_embeddings  # (N_patches, 768)
gt_attn = softmax(cosine_sim(sentence_emb, patch_emb) / temp)
```

**(b) CAMELYON16 explicit supervision (eğer eğitim setine dahil edilirse):**
```python
# Pixel-level tumor mask → patch-level binary
gt_attn = patch_tumor_mask / patch_tumor_mask.sum()  # normalized
```

**Loss formülasyonu:** KL divergence (aşağı yukarı [Direct Visual Grounding, 2511.12738]'in formülasyonuna paralel)

```python
def grounding_loss(generated_attn, gt_attn):
    # generated_attn: (N_sentences, N_vision_tokens)
    # gt_attn: (N_sentences, N_vision_tokens)
    return F.kl_div(F.log_softmax(generated_attn, dim=-1),
                    F.softmax(gt_attn, dim=-1),
                    reduction='batchmean')
```

**Attention extraction — kritik teknik nokta:**
LLaMA decoder-only model. "Decoder cross-attention" YOK; vision token'lar text token'larla aynı sequence'a girer ve causal self-attention üzerinden işlenir. Extraction:
```python
# transformers'tan output_attentions=True ile attention weights
attentions = outputs.attentions  # tuple of (B, n_heads, seq_len, seq_len)
# Belirli layer ve head'lerden text→vision attention slice
text_to_vision = attentions[layer_idx][:, head_idx, text_pos, vision_pos_start:vision_pos_end]
```

LLaVA literatüründe (Kang et al. CVPR 2025) layer 14 head 24 gibi spesifik head'lerin grounding-faithful olduğu gösterilmiş. Bizim ablation'umuzda hangi layer/head'in en iyi grounding sinyali verdiğini ölçmemiz gerekecek.

### L_faithfulness — Attention Concentration Penalty

ACMIL'in branch concentration loss'undan adapte:

```python
def faithfulness_reg(sentence_attentions):
    # Her cümle attention'ının entropy'sini minimize et
    # → cümle başına az patch'e odaklanma
    entropy = -(attentions * torch.log(attentions + 1e-8)).sum(dim=-1)
    return entropy.mean()
```

## Hyperparameter (Başlangıç)

```yaml
optimizer: AdamW
lr_adapter: 1e-4
lr_lora: 5e-5
weight_decay: 0.01
warmup_steps: 100
scheduler: cosine
batch_size: 1
gradient_accumulation: 16
epochs: 3-5
mixed_precision: bf16
gradient_checkpointing: true

# Loss weights
lambda_grounding: 0.3
lambda_faithfulness: 0.1

# LoRA
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

## Eğitim Hesabı

| Aşama | VRAM (peak) | Süre (A6000 48GB) |
|-------|-------------|-------------------|
| CONCHv1.5 + TITAN embedding precompute (150 slide) | 24 GB | 8-12 saat |
| Caption baseline (3 epoch) | 28 GB | 10-14 saat |
| Grounding loss v1 (3 epoch) | 32 GB | 14-18 saat |
| Grounding loss v2 (3 epoch) | 32 GB | 14-18 saat |
| Evaluation (full suite) | 16 GB | 4-6 saat |

**Toplam:** ~50-70 GPU saat → **$22-31 RunPod bütçesi**.
Buffer ile **$35-40** ayrılması önerilir.

**Storage:**
- TCGA-BRCA WSI (RunPod): ~250 GB
- CAMELYON16 (RunPod): ~150 GB
- Embedding HDF5 (lokal'e sync): ~30 GB
- Checkpoints: ~5 GB

## Risk Noktaları ve Açık Sorular

**Risk 1 — TITAN slide encoder API'si:**
TITAN HuggingFace `trust_remote_code=True` ile yükleniyor. Slide encoder'ı doğrudan exposed mi, yoksa TITAN.encode_slide() yüksek-seviye API'si mi var? Hafta 2 başında doğrulanmalı. Plan B: TITAN tamamı kullanılır (text encoder dahil), slide tokens TITAN'ın internal forward'undan extract edilir.

**Risk 2 — Vision token sayısı:**
TITAN'ın çıkardığı slide_tokens sayısı slide'a göre değişiyor (patch grid'e bağlı). Llama context window 128K, sorun değil ama batch padding stratejisi gerekli. Çözüm: max_vision_tokens=1024, fazlası attention pooling ile compress.

**Risk 3 — Attention extraction layer/head seçimi:**
Hangi LLM layer'ın grounding-faithful olduğu Llama-3.2-3B'de bilinmiyor (LLaVA-1.5 7B için layer 14 raporlanmış). Hafta 3'te 5-10 örnek üzerinde her layer'ın attention map'lerini görsel kontrol et, en faithful layer'ı seç.

**Risk 4 — Grounding loss tasarımı:**
KL divergence baseline; eğer iyileşme yoksa MSE veya cosine alignment dene. v1 vs v2 ablation'ında bu test edilir.

**Risk 5 — TCGA pathology raporları noisy:**
SlideInstruction zaten GPT-4 ile yapılandırılmış, bu sorunu büyük ölçüde çözüyor. Doğrudan TCGA PDF parsing yapmıyoruz.

## Açık Tasarım Soruları (Hafta 1-2'de cevaplanacak)

- [ ] TITAN slide encoder forward'u nasıl çağrılıyor? (HF model card'a bakılacak)
- [ ] Grounding loss için optimal layer/head LLama-3.2-3B'de hangi?
- [ ] Pseudo-grounding cosine threshold ne olmalı (0.3? 0.5?)?
- [ ] CAMELYON16'yı eğitime dahil edersek (v2'de) cross-dataset overfitting riski var mı?

## Referanslar

- TITAN (Mahmood Lab, 2024): https://arxiv.org/abs/2411.19666
- CONCH (Nature Medicine 2024): https://www.nature.com/articles/s41591-024-02856-4
- CONCHv1.5 (HF model card): https://huggingface.co/MahmoodLab/conchv1_5
- SlideChat (CVPR 2025, referans): https://arxiv.org/abs/2410.11761
- LLaVA-1.5 (NeurIPS 2023): https://arxiv.org/abs/2310.03744
- Direct Visual Grounding via KL Attention Loss (2025): https://arxiv.org/abs/2511.12738
- MedGround (Jan 2026): https://arxiv.org/abs/2601.06847
- "Few Attention Heads Suffice" (Kang et al. CVPR 2025): grounding-faithful head identification
