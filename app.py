"""
SatQuery AI - interface.

Deliberately plain everywhere except the evidence overlay, which is the
feature the pitch rests on and therefore gets the visual care.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image

import config
from demo.samples import SCENES, build as build_scene
from pipeline import build_pipeline
from render import render_answer

st.set_page_config(page_title="SatQuery AI", layout="wide")

EXAMPLES = [
    "What land cover is in this image?",
    "Describe this scene.",
    "Where is the water?",
    "What changed between these two images?",
    "How much of this tile is urban?",
]


@st.cache_resource
def get_pipeline(with_vlm: bool):
    """Cached so weights load once per session, not once per query."""
    return build_pipeline(with_vlm=with_vlm)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.subheader("System")
    use_vlm = st.toggle(
        "Use fine-tuned VLM",
        value=bool(config.VLM_ADAPTER),
        help="Off runs the heuristic baselines only. Useful for showing the "
             "before/after side by side in a demo.",
    )
    overlay_mode = st.selectbox(
        "Evidence overlay", ["auto", "change", "heatmap", "regions"],
        help="auto picks the right view for the query.",
    )

pipeline = get_pipeline(use_vlm)

with st.sidebar:
    st.caption(f"Parser: `{getattr(pipeline.parser, 'parser_id', 'unknown')}`")
    st.caption(f"LLM backend: `{config.LLM_BACKEND}`")

    st.subheader("Registered models")
    st.caption("Adding a capability means registering a model, not editing "
               "routing logic.")
    for m in pipeline.registry.describe():
        flag = " · loaded" if m["loaded"] else ""
        st.markdown(f"**{m['model_id']}**  \n{', '.join(m['tasks'])} · "
                    f"priority {m['priority']}{flag}")

# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------

st.title("SatQuery AI")
st.caption("Ask a question about satellite imagery in plain English. "
           "Every answer names the model that produced it and points at "
           "the region it came from.")

source = st.radio("Imagery", ["Upload", "Sample scene"], horizontal=True,
                  label_visibility="collapsed")

images: list[Image.Image] = []
default_query = ""

if source == "Upload":
    uploaded = st.file_uploader(
        "Satellite images",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        help="Upload one image, or two of the same area from different dates "
             "to ask what changed.",
    )
    for f in uploaded or []:
        images.append(Image.open(io.BytesIO(f.read())))
else:
    scene = st.selectbox("Scene", list(SCENES))
    before, after, _, desc = build_scene(scene, 384)
    images = [before, after]
    default_query = "What changed between these two images?"
    st.caption(f"Synthetic scene — {desc}. Works with no network or dataset.")

if images:
    cols = st.columns(min(len(images), 4))
    for i, (col, im) in enumerate(zip(cols, images)):
        col.image(im, caption=f"Image {i + 1} · {im.size[0]}x{im.size[1]}",
                  use_container_width=True)
else:
    st.info("Upload an image to start. No image, no answer — the system will "
            "not guess.")

query = st.text_input("Question", value=default_query,
                      placeholder=EXAMPLES[0])
st.caption("Try: " + " · ".join(f"`{e}`" for e in EXAMPLES))

# --------------------------------------------------------------------------
# Answer
# --------------------------------------------------------------------------

if st.button("Ask", type="primary", disabled=not query):
    answer = pipeline.ask(query, images)

    if answer.spec:
        s = answer.spec
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Task", s.task_type.value)
        c2.metric("Modality", s.modality.value)
        c3.metric("Parser confidence", f"{s.confidence:.0%}")
        c4.metric("Latency", f"{answer.total_latency_ms:.0f} ms")

    # -- the plan, shown before the answer -------------------------------
    if answer.plan and answer.plan.model_ids:
        with st.expander(
                f"Execution plan — {len(answer.plan)} model(s), "
                f"{answer.plan.fusion.value} fusion", expanded=False):
            st.write(answer.plan.rationale)
            st.code(" -> ".join(answer.plan.model_ids), language=None)

    if answer.answered:
        st.success(answer.text)
        for out in answer.outputs:
            st.caption(f"`{out.model_id}` · confidence {out.confidence:.0%} · "
                       f"{out.latency_ms:.0f} ms"
                       + (" · VETO" if out.veto else ""))
            if out.scores:
                st.bar_chart(out.scores)
    else:
        st.warning(answer.text)
        st.caption("This is a refusal, not a failure. The system declines when "
                   "the inputs cannot support the question.")

    # -- evidence ---------------------------------------------------------
    rendered = render_answer(images, answer, mode=overlay_mode)
    if rendered:
        st.subheader("Evidence")
        for caption, img in rendered.items():
            st.image(img, caption=caption, use_container_width=True)

    notes = [e for e in answer.evidence if e.bbox is None and e.note]
    for e in notes:
        st.caption(f"Image {e.image_index + 1}: {e.note}")
