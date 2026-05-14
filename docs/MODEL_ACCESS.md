# HuggingFace Gated Model Access

PathoLens uses three gated HuggingFace models. You must request access **before** running any training or embedding pipeline.

## Prerequisites

- A HuggingFace account with an **institutional email** (e.g., `@itu.edu.tr`)
- `@gmail`, `@hotmail`, etc. will be **denied** gated access
- Access token: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## Models to Request

### 1. CONCHv1.5 (Patch Encoder)

- **Repo:** [MahmoodLab/conchv1_5](https://huggingface.co/MahmoodLab/conchv1_5)
- **License:** CC-BY-NC-ND 4.0 (research only)
- **Action:** Click "Access repository" → fill form → typically approved within 24h
- **Used for:** Patch-level feature extraction (448×448 → 768-dim embedding)

### 1b. CONCH v1 (Pseudo-GT Encoder)

- **Repo:** [MahmoodLab/conch](https://huggingface.co/MahmoodLab/conch)
- **License:** CC-BY-NC-ND 4.0 (research only)
- **Action:** Click "Access repository" → fill form
- **Used for:** Pseudo-grounding GT generation (text encoder + vision encoder, shared 512-dim space)

**Intended Research Use (form'a yaz):**

> We are conducting a capstone project on grounded report generation for breast cancer histopathology at Istanbul Technical University, Department of AI and Data Engineering, supervised by Prof. Dr. Behçet Uğur Töreyin. We use CONCH v1 exclusively to generate pseudo-ground-truth attention supervision for a grounding loss: for each sentence in a generated pathology report, we compute cosine similarity between the CONCH text embedding of the sentence and the CONCH patch embeddings of the corresponding whole-slide image to produce a soft attention distribution over patches. This signal supervises a Llama-3.2-3B model to produce spatially faithful reports. The model is used only in research and will not be deployed commercially. No patient data leaves our secure compute environment.

> **Note:** We also request CONCHv1.5 access (vision-only) for patch feature extraction compatible with the TITAN slide encoder. CONCH v1 is needed separately because CONCHv1.5 does not include a text encoder.

### 2. TITAN (Slide Encoder)

- **Repo:** [MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN)
- **License:** CC-BY-NC-ND 4.0 (research only)
- **Action:** Click "Access repository" → fill form
- **Used for:** Slide-level aggregation (patch embeddings → slide tokens)

### 3. Llama-3.2-3B-Instruct (Language Model)

- **Repo:** [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- **License:** Meta Llama 3.2 Community License
- **Action:** Click "Access repository" → accept Meta's license agreement
- **Used for:** Report generation with LoRA fine-tuning

## After Approval

1. Generate a **read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Add to your `.env`:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```
3. Verify access:
   ```bash
   uv run huggingface-cli whoami
   ```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `401 Unauthorized` | Token invalid or not set in `.env` |
| `403 Forbidden` | Access not yet approved — check email |
| `gated repo` error | Re-request access with institutional email |
