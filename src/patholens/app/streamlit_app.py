"""PathoLens-VLM Streamlit demo.

Pick one of the precomputed demo slides → generate a structured pathology
report → inspect per-sentence attention overlays on the slide thumbnail →
download a FHIR DiagnosticReport JSON.

Run on the machine that has the checkpoint + GPU + H5 files:

    uv run streamlit run src/patholens/app/streamlit_app.py \\
        --server.port 8501 --server.address 0.0.0.0

…then SSH-tunnel to your laptop:

    ssh -L 8501:localhost:8501 root@<runpod-ip> -p <port> -i ~/.ssh/id_ed25519 -N
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import streamlit as st
import torch

from patholens.app.inference import DEFAULT_INSTRUCTION, generate_with_attention
from patholens.app.overlay import attention_to_heatmap, overlay_on_thumbnail
from patholens.models.grounded_vlm import GroundedVLM
from patholens.reporting.fhir import generated_text_to_fhir_diagnostic_report

DEFAULT_CKPT = os.environ.get(
    "PATHOLENS_CKPT",
    "checkpoints/grounded_smoke_20260602_105250/final",
)
DEFAULT_EMB_DIR = os.environ.get(
    "PATHOLENS_EMB_DIR",
    "data/processed/embeddings/tcga_brca",
)
DEFAULT_MAX_TOKENS = 1024
DEFAULT_PATCH_SIZE_L0 = 896  # 448 × 2 (20× over 40× native)

st.set_page_config(page_title="PathoLens-VLM Demo", layout="wide")


@st.cache_resource(show_spinner="Loading checkpoint (one-off ~30 s)…")
def load_model(checkpoint_dir: str) -> GroundedVLM:
    model = GroundedVLM.from_pretrained(checkpoint_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.adapter = model.adapter.to(device)
    if model.llm is not None:
        model.llm = model.llm.to(device)
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def load_slide(h5_path: str, max_tokens: int) -> dict:
    with h5py.File(h5_path, "r") as f:
        emb = f["patch_embeddings_v15"][:]
        coords = f["coordinates"][:]
        mask = f["tissue_mask_lowres"][:]
        attrs = dict(f.attrs)
    if emb.shape[0] > max_tokens:
        idx = np.linspace(0, emb.shape[0] - 1, max_tokens).astype(np.int64)
        emb = emb[idx]
        coords = coords[idx]
    return {
        "embeddings": emb,
        "coordinates": coords,
        "mask_lowres": mask,
        "attrs": attrs,
        "n_patches_total": int(attrs.get("total_tissue_patches", emb.shape[0])),
    }


def _list_slides(emb_dir: Path) -> list[str]:
    return sorted(p.stem for p in emb_dir.glob("*.h5"))


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("PathoLens-VLM")
st.sidebar.caption("Spatially-grounded WSI report generation — capstone demo")

emb_dir = Path(st.sidebar.text_input("Embeddings dir", DEFAULT_EMB_DIR))
ckpt_dir = st.sidebar.text_input("Checkpoint", DEFAULT_CKPT)

slides = _list_slides(emb_dir) if emb_dir.exists() else []
if not slides:
    st.sidebar.error(f"No .h5 found under {emb_dir}")
    st.stop()

slide_id = st.sidebar.selectbox("Slide", slides, index=0)
instruction = st.sidebar.text_area("Instruction", DEFAULT_INSTRUCTION, height=110)
max_new = st.sidebar.slider("max new tokens", 64, 768, 384, step=32)
go = st.sidebar.button("Generate", type="primary", use_container_width=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Grounded pathology report")
st.caption(
    f"Slide **{slide_id}** · checkpoint `{Path(ckpt_dir).name}` · "
    f"vision-token cap {DEFAULT_MAX_TOKENS}"
)

# Load model up-front (cached) so the button is responsive
model = load_model(ckpt_dir)
device = next(model.adapter.parameters()).device

slide = load_slide(str(emb_dir / f"{slide_id}.h5"), DEFAULT_MAX_TOKENS)

if "result" not in st.session_state:
    st.session_state["result"] = None
    st.session_state["result_slide"] = None

if go:
    vision = torch.from_numpy(slide["embeddings"]).to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    with st.spinner("Generating + extracting attention…"):
        st.session_state["result"] = generate_with_attention(
            model, vision_tokens=vision, instruction=instruction, max_new_tokens=max_new,
        )
        st.session_state["result_slide"] = slide_id

result = st.session_state["result"]
if result is None or st.session_state["result_slide"] != slide_id:
    st.info("Pick a slide on the left and click **Generate**.")
    st.stop()

tabs = st.tabs(["Report", "Grounding", "FHIR", "Debug"])

# ── Report ───────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### Generated report")
    st.write(result["text"])
    st.divider()
    st.markdown(f"**Sentences:** {len(result['sentences'])}  ·  "
                f"**vision tokens:** {result['n_vision_tokens']}")

# ── Grounding ────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### Per-sentence attention")
    sent_options = [
        f"{i + 1}. {s[:80] + ('…' if len(s) > 80 else '')}"
        for i, s in enumerate(result["sentences"])
    ]
    sel = st.radio("Sentence", sent_options, index=0, key=f"sel_{slide_id}")
    sel_idx = sent_options.index(sel)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("**Selected sentence**")
        st.info(result["sentences"][sel_idx])
    with col_b:
        attn = result["per_sentence_attn"][sel_idx]
        st.markdown("**Attention concentration**")
        # quick concentration summary
        sorted_a = np.sort(attn)[::-1]
        top10_mass = float(sorted_a[: max(1, len(attn) // 10)].sum())
        st.metric("Top-10% patch mass", f"{top10_mass:.1%}")

    heat = attention_to_heatmap(
        attn=attn,
        coordinates=slide["coordinates"],
        mask_lowres=slide["mask_lowres"],
        patch_size_l0=int(slide["attrs"].get("patch_size", 448)) * 2,
    )
    img = overlay_on_thumbnail(slide["mask_lowres"], heat)
    st.image(img, caption=f"Attention overlay for sentence {sel_idx + 1}",
             use_container_width=True)

# ── FHIR ─────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### HL7 FHIR DiagnosticReport")
    fhir = generated_text_to_fhir_diagnostic_report(
        generated_text=result["text"],
        slide_id=slide_id,
    )
    fhir_json = json.dumps(fhir, indent=2, default=str)
    st.code(fhir_json, language="json")
    st.download_button(
        "Download .json",
        data=fhir_json,
        file_name=f"{slide_id}_fhir.json",
        mime="application/json",
    )

# ── Debug ────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### Debug")
    st.json({
        "slide_id": slide_id,
        "n_patches_kept": int(slide["embeddings"].shape[0]),
        "n_patches_total": slide["n_patches_total"],
        "n_vision_tokens": result["n_vision_tokens"],
        "n_sentences": len(result["sentences"]),
        "checkpoint": ckpt_dir,
        "device": str(device),
        "mask_shape": list(slide["mask_lowres"].shape),
    })
