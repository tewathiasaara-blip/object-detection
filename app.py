"""
app.py  --  Streamlit interface (optional module, run AFTER the core works)
===========================================================================
A clean upload-and-view web interface for the object detector. It is a thin
wrapper over the SAME detection code used on the command line (src/common.py),
so the results are identical -- the UI just makes them easy to show in a seminar.

Run it with:
    streamlit run app.py

Then open the URL it prints (usually http://localhost:8501).
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.common import (
    load_model, pick_device, run_on_image,
    DEFAULT_WEIGHTS, DEFAULT_CONF, DEFAULT_IMGSZ,
)

# ---------------------------------------------------------------------------
# Page configuration and light styling.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Object Detection — Edge AI Demo",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      .metric-card {background:#0e1117; border:1px solid #2a2f3a; border-radius:12px;
                    padding:14px 18px;}
      footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load the model once and cache it (so it is not reloaded on every click).
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO model...")
def get_model(weights: str):
    return load_model(weights)


# ---------------------------------------------------------------------------
# Sidebar: settings. These are REAL knobs that change the detection.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    weights = st.selectbox(
        "Model weights",
        options=["yolo11n.pt", "yolo11s.pt", "yolov8n.pt"],
        index=0,
        help="Nano (n) is fastest on M1. 's' is a bit more accurate but slower.",
    )
    conf = st.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05,
                    help="Detections below this confidence are hidden.")
    imgsz = st.select_slider("Inference image size", options=[320, 480, 640, 960],
                            value=DEFAULT_IMGSZ,
                            help="Smaller = faster, larger = can catch small objects.")
    device = pick_device()
    st.caption(f"Compute device: **{device.upper()}**  "
            f"({'Apple GPU' if device == 'mps' else 'CPU'})")
    st.divider()
    st.caption("This demo runs a **pretrained** YOLO model (COCO, 80 classes). "
            "It was not trained from scratch.")


# ---------------------------------------------------------------------------
# Main area.
# ---------------------------------------------------------------------------
st.title("🎯 Real-Time Object Detection & Classification")
st.write("Upload an image to detect and locate common objects using a deep-learning "
        "YOLO model running on Apple Silicon.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    # Decode the uploaded bytes into an OpenCV BGR image.
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not read that file as an image.")
    else:
        model = get_model(weights)
        t0 = time.time()
        result = run_on_image(model, img, source_name=uploaded.name,
                            conf=conf, imgsz=imgsz, device=device)
        wall_ms = (time.time() - t0) * 1000.0

        # Convert BGR (OpenCV) to RGB (Streamlit) for correct colours.
        original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotated_rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(original_rgb, use_container_width=True)
        with col2:
            st.subheader("Detected")
            st.image(annotated_rgb, use_container_width=True)

        # Honest metric row.
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Objects detected", result.count())
        m2.metric("Inference time", f"{result.inference_ms:.0f} ms")
        m3.metric("Total model time", f"{result.total_ms:.0f} ms")
        m4.metric("Wall-clock", f"{wall_ms:.0f} ms")

        # Detection table.
        if result.count() > 0:
            rows = [{
                "class": d.class_name,
                "confidence": round(d.confidence, 3),
                "x1": int(d.x1), "y1": int(d.y1),
                "x2": int(d.x2), "y2": int(d.y2),
            } for d in result.detections]
            df = pd.DataFrame(rows).sort_values("confidence", ascending=False)
            st.subheader("Detections")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Let the user download the annotated image.
            ok, buf = cv2.imencode(".jpg", result.annotated_image)
            if ok:
                st.download_button("⬇️ Download annotated image", buf.tobytes(),
                                file_name=f"{os.path.splitext(uploaded.name)[0]}_detected.jpg",
                                mime="image/jpeg")
        else:
            st.info("No objects were detected above the current confidence threshold. "
                    "Try lowering it in the sidebar.")
else:
    st.info("👆 Upload an image to begin. Tip: a street scene or a desk with "
            "everyday objects shows the detector well.")
