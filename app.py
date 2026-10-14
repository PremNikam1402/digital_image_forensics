"""Digital Image Forensics and Tampering Detection System - Streamlit Dashboard.

A systematic, production-grade multi-technique digital forensic workbench featuring:
- Cryptographic Integrity & SHA-256 Hashing
- EXIF & Structural Metadata Forensics (ExifTool / Pillow)
- Error Level Analysis (ELA) with Interactive Quality Tuning
- High-Frequency Sensor Noise & Local Variance Inconsistency
- Histogram Moments, Shannon Entropy & Comb Anomaly Detection
- SIFT/ORB Copy-Move Duplicate Region Forgery Detection
- Unified 24-Dimensional Forensic Feature Vector Inspection
- Supervised Random Forest Classification & Calibrated Confidence
- Automated Court-Ready PDF & Markdown Forensic Report Generation
"""

from __future__ import annotations

import io
import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Ensure UTF-8 stdout encoding in Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Forensic core modules
from forensic.preprocessing import (
    validate_image,
    calculate_sha256,
    get_image_properties,
    to_rgb_array,
)
from forensic.metadata import extract_metadata
from forensic.ela import perform_ela
from forensic.noise_analysis import perform_noise_analysis
from forensic.histogram import perform_histogram_analysis
from forensic.copy_move import detect_copy_move
from forensic.feature_extraction import (
    extract_all_features,
    FEATURE_NAMES,
    FEATURE_DESCRIPTIONS,
)
from ml.predict import ForensicPredictor
from utils.visualization import (
    plot_histogram,
    plot_ela_comparison,
    plot_noise_map,
    plot_copy_move_matches,
    plot_confusion_matrix,
)
from utils.report_generator import generate_forensic_report

# Absolute path anchor for reliable cross-platform execution
BASE_DIR = Path(__file__).resolve().parent


# ==========================================
# PAGE CONFIGURATION & MODERN STYLING
# ==========================================
st.set_page_config(
    page_title="Digital Image Forensics System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Global Typography & Header */
    .app-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #10b981;
        font-weight: 600;
        margin-bottom: 1.2rem;
    }
    .header-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        background-color: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin-bottom: 0.6rem;
    }

    /* Executive Verdict Cards */
    .verdict-card-manipulated {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.16), rgba(185, 28, 28, 0.08));
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.15);
    }
    .verdict-card-authentic {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.16), rgba(5, 150, 105, 0.08));
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.15);
    }
    .verdict-title {
        font-size: 0.85rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .verdict-heading {
        font-size: 2.2rem;
        font-weight: 900;
        margin: 0;
        line-height: 1.15;
    }

    /* Pill Badges */
    .forensic-pill {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 8px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background: rgba(128, 128, 128, 0.08);
        word-break: break-word;
    }

    /* Hash Box */
    .hash-container {
        font-family: 'Consolas', 'Courier New', monospace;
        background-color: rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(128, 128, 128, 0.3);
        padding: 10px 14px;
        border-radius: 6px;
        word-break: break-all;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# INITIALIZATION & MODEL CACHING
# ==========================================
@st.cache_resource
def get_predictor() -> ForensicPredictor:
    """Load cached inference model."""
    return ForensicPredictor()


predictor = get_predictor()


# ==========================================
# SIDEBAR: CONTROLS & SETTINGS
# ==========================================
with st.sidebar:
    st.markdown("### 🛡️ Forensic Workbench")
    st.caption("BE Capstone Project • Digital Forensics System")

    # Image upload input
    uploaded_file = st.file_uploader(
        "Upload Evidence Image",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"],
        help="Upload JPG, JPEG, PNG, or TIFF image to examine.",
    )

    # Preloaded sample loader
    st.markdown("#### Or Select Pre-loaded Sample:")
    sample_options = [
        "None (Use Uploaded Image)",
        "📸 Authentic Camera Photograph (Sample 1)",
        "📸 Authentic Camera Photograph (Sample 2)",
        "✂️ Copy-Move Forgery Sample",
        "🧩 Spliced Composite Sample",
        "📊 Contrast Manipulated Sample",
    ]

    # Sync with session_state if set from landing page
    current_default_index = 0
    if "selected_sample_label" in st.session_state:
        if st.session_state["selected_sample_label"] in sample_options:
            current_default_index = sample_options.index(st.session_state["selected_sample_label"])

    sample_choice = st.selectbox(
        "Quick Test Samples:",
        options=sample_options,
        index=current_default_index,
        key="sidebar_sample_select",
    )

    st.markdown("---")
    with st.expander("⚙️ Advanced Analysis Settings", expanded=False):
        ela_quality = st.slider(
            "ELA JPEG Quality",
            min_value=60,
            max_value=98,
            value=90,
            step=1,
            help="Compression quality for Error Level Analysis comparison (Standard: 90-95%)",
        )

        copy_move_algo = st.selectbox(
            "Copy-Move Detector",
            options=["SIFT", "ORB"],
            index=0,
            help="SIFT provides superior scale & rotation invariance; ORB is fast binary matching.",
        )

        lowe_ratio = st.slider(
            "Lowe's Ratio Threshold",
            min_value=0.50,
            max_value=0.90,
            value=0.75,
            step=0.05,
            help="Ratio threshold for nearest-to-second-nearest neighbor distance.",
        )

        min_spatial_dist = st.slider(
            "Min Spatial Distance (px)",
            min_value=15,
            max_value=100,
            value=35,
            step=5,
            help="Minimum Euclidean pixel distance between matched keypoints to exclude self-matches.",
        )

    run_analysis_btn = st.button("🚀 Run Forensic Analysis", type="primary", width="stretch")

    st.markdown("---")
    st.markdown("#### 🤖 Classifier Engine Status")
    if predictor.model_loaded and predictor.metrics:
        acc = predictor.metrics.get("accuracy", 0.0) * 100
        f1 = predictor.metrics.get("f1_score", 0.0)
        n_samples = predictor.metrics.get("train_samples", 0) + predictor.metrics.get("test_samples", 0)
        st.success(
            f"**Random Forest Classifier**\n"
            f"- Accuracy: **{acc:.1f}%**\n"
            f"- F1-Score: **{f1:.3f}**\n"
            f"- Training Dataset: **{n_samples} samples**"
        )
    elif predictor.model_loaded:
        st.success("**Random Forest Classifier**\n- Status: Loaded")
    else:
        st.warning("**Classifier**: Model file pending\n(Using conservative rule integrator)")


# ==========================================
# TOP HEADER
# ==========================================
st.markdown("<span class='header-badge'>Forensic Science & Cybersecurity</span>", unsafe_allow_html=True)
st.markdown("<div class='app-title'>Digital Image Forensics & Tampering Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Multi-Technique Digital Forensic Examination Workbench • BE Project</div>", unsafe_allow_html=True)


# ==========================================
# DATA LOADING & SELECTION LOGIC
# ==========================================
target_image_bytes: Optional[bytes] = None
target_image_filename: Optional[str] = None

# Reliable absolute paths to test samples
sample_map = {
    "📸 Authentic Camera Photograph (Sample 1)": BASE_DIR / "data" / "sample" / "authentic" / "authentic_001.jpg",
    "📸 Authentic Camera Photograph (Sample 2)": BASE_DIR / "data" / "sample" / "authentic" / "authentic_002.jpg",
    "✂️ Copy-Move Forgery Sample": BASE_DIR / "data" / "sample" / "tampered" / "tampered_001.jpg",
    "🧩 Spliced Composite Sample": BASE_DIR / "data" / "sample" / "tampered" / "tampered_002.jpg",
    "📊 Contrast Manipulated Sample": BASE_DIR / "data" / "sample" / "tampered" / "tampered_003.jpg",
}

if sample_choice != "None (Use Uploaded Image)" and sample_choice in sample_map:
    sample_path = sample_map[sample_choice]
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            target_image_bytes = f.read()
        target_image_filename = sample_path.name
    else:
        st.error(f"Sample file not found at: {sample_path}")

elif uploaded_file is not None:
    target_image_bytes = uploaded_file.getvalue()
    target_image_filename = uploaded_file.name


# ==========================================
# VIEW 1: LANDING WORKSPACE (NO IMAGE YET)
# ==========================================
if target_image_bytes is None:
    st.info("👋 **Welcome to the Digital Image Forensics System.** Please upload an evidence image or select a verified test case below to begin forensic analysis.")

    # 3-Step Systematic Process
    st.markdown("### 📋 Systematic Examination Methodology")
    step1, step2, step3 = st.columns(3)
    with step1:
        with st.container(border=True):
            st.markdown("#### 1. Ingestion & Custody")
            st.markdown(
                "- Cryptographic SHA-256 byte hashing\n"
                "- File integrity & byte stream validation\n"
                "- Hardware EXIF & camera optical tags\n"
                "- Software signature detection (Photoshop/GIMP)"
            )
    with step2:
        with st.container(border=True):
            st.markdown("#### 2. Physical & Signal Forensics")
            st.markdown(
                "- Error Level Analysis (ELA) re-compression\n"
                "- Sensor noise residual & local variance\n"
                "- Shannon entropy & comb anomaly detection\n"
                "- SIFT / ORB copy-move clone clustering"
            )
    with step3:
        with st.container(border=True):
            st.markdown("#### 3. Classification & Report")
            st.markdown(
                "- 24-dimensional feature vector extraction\n"
                "- Supervised Random Forest classification\n"
                "- Calibrated class probabilities\n"
                "- Certified PDF & Markdown examination dossier"
            )

    st.markdown("---")
    st.markdown("### ⚡ Quick Test Cases (Click to Load)")
    st.write("You can test the system immediately with verified authentic and forged forensic benchmarks:")

    col_demo1, col_demo2, col_demo3 = st.columns(3)
    with col_demo1:
        with st.container(border=True):
            st.markdown("#### 📸 Authentic DSLR Sample")
            st.caption("Pristine camera capture with intact EXIF tags, uniform single-cycle JPEG compression, and homogeneous sensor noise floor.")
            if st.button("Load Authentic Sample 1", key="btn_auth1", width="stretch"):
                st.session_state["selected_sample_label"] = "📸 Authentic Camera Photograph (Sample 1)"
                st.rerun()
            if st.button("Load Authentic Sample 2", key="btn_auth2", width="stretch"):
                st.session_state["selected_sample_label"] = "📸 Authentic Camera Photograph (Sample 2)"
                st.rerun()

    with col_demo2:
        with st.container(border=True):
            st.markdown("#### ✂️ Copy-Move Forgery")
            st.caption("Contains cloned duplicate regions. SIFT/ORB feature matching with RANSAC affine verification detects duplicate keypoint clusters.")
            if st.button("Load Copy-Move Sample", key="btn_cm", width="stretch", type="primary"):
                st.session_state["selected_sample_label"] = "✂️ Copy-Move Forgery Sample"
                st.rerun()

    with col_demo3:
        with st.container(border=True):
            st.markdown("#### 🧩 Splicing & Enhancement")
            st.caption("Composite image exhibiting mismatched high-frequency sensor noise profiles, ELA boundary discrepancies, and comb gaps.")
            if st.button("Load Spliced Sample", key="btn_splice", width="stretch"):
                st.session_state["selected_sample_label"] = "🧩 Spliced Composite Sample"
                st.rerun()
            if st.button("Load Contrast Sample", key="btn_contrast", width="stretch"):
                st.session_state["selected_sample_label"] = "📊 Contrast Manipulated Sample"
                st.rerun()

    st.stop()


# ==========================================
# VIEW 2: ACTIVE FORENSIC EXAMINATION
# ==========================================
else:
    # Save to temporary file to preserve raw byte integrity for EXIF and hashing
    temp_dir = tempfile.mkdtemp()
    temp_image_path = Path(temp_dir) / (target_image_filename or "evidence.jpg")
    with open(temp_image_path, "wb") as f:
        f.write(target_image_bytes)

    # Step 1: Validate Image Integrity
    is_valid, val_msg, pil_img = validate_image(temp_image_path)
    if not is_valid or pil_img is None:
        st.error(f"❌ Image Integrity Validation Failed: {val_msg}")
        st.stop()

    # Step 2: Compute Structural Properties & Hashes
    props = get_image_properties(pil_img, raw_bytes=target_image_bytes, filename=target_image_filename)

    # Step 3: Run / Retrieve Cached Analysis
    cache_key = f"forensic_{props['sha256']}_{ela_quality}_{copy_move_algo}_{lowe_ratio}_{min_spatial_dist}"

    if cache_key not in st.session_state or run_analysis_btn:
        with st.spinner("Executing multi-technique digital forensic examination..."):
            pred_results = predictor.predict(
                image=pil_img,
                image_path=temp_image_path,
                copy_move_algo=copy_move_algo,
                ela_quality=ela_quality,
                lowe_ratio=lowe_ratio,
                min_spatial_distance=min_spatial_dist,
            )
            report_bundle = generate_forensic_report(pred_results, props)

            st.session_state[cache_key] = {
                "results": pred_results,
                "report": report_bundle,
            }

    analysis_data = st.session_state[cache_key]
    pred_res = analysis_data["results"]
    report_bundle = analysis_data["report"]
    full_res = pred_res["full_results"]

    is_tamp = pred_res["is_tampered"]
    pred_label = pred_res["prediction"]
    conf_pct = pred_res["confidence_percent"]

    ela_stats = full_res["ela"]["stats"]
    noise_stats = full_res["noise"]["stats"]
    cm_res = full_res["copy_move"]["results"]
    meta = full_res["metadata"]
    hist_stats = full_res["histogram"]


    # ==========================================
    # TOP EXECUTIVE FORENSIC VERDICT BANNER
    # ==========================================
    verdict_card_class = "verdict-card-manipulated" if is_tamp else "verdict-card-authentic"
    verdict_color = "#ef4444" if is_tamp else "#10b981"
    verdict_icon = "🚨" if is_tamp else "✅"

    # Primary physical reason summary
    if cm_res["evidence_found"]:
        reason_summary = f"Definitive physical evidence: {cm_res['inliers_count']} RANSAC-verified duplicate keypoint pairs detected via {cm_res['algorithm']}."
    elif ela_stats["suspicious_ela"] and noise_stats["suspicious_noise"]:
        reason_summary = "High compression variance across blocks and discordant local sensor noise variance indicate multi-source splicing."
    elif ela_stats["suspicious_ela"]:
        reason_summary = f"Significant ELA block error discrepancy ({ela_stats['block_discrepancy_ratio']:.2f}x) observed between regions."
    elif noise_stats["suspicious_noise"]:
        reason_summary = f"Abnormal sensor noise variance ({noise_stats['inconsistency_ratio']*100:.1f}% inconsistent blocks) detected."
    elif is_tamp:
        reason_summary = "Non-linear statistical anomalies and structural feature signatures match tampering distributions."
    else:
        reason_summary = "Homogeneous compression level, consistent sensor noise floor, and intact single-cycle JPEG characteristics."

    st.markdown(
        f"""
        <div class='{verdict_card_class}'>
            <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;'>
                <div style='flex: 1; min-width: 300px;'>
                    <div class='verdict-title' style='color: {verdict_color};'>
                        {verdict_icon} Automated Forensic Decision
                    </div>
                    <div class='verdict-heading' style='color: {verdict_color};'>
                        {pred_label}
                    </div>
                    <div style='font-size: 0.95rem; margin-top: 6px;'>
                        <b>Primary Finding:</b> {reason_summary}
                    </div>
                    <div style='font-size: 0.82rem; margin-top: 4px; opacity: 0.85;'>
                        Engine: <b>{pred_res['source']}</b> • Evidence Sample: <code>{props['filename']}</code>
                    </div>
                </div>
                <div style='text-align: right; min-width: 220px; margin-top: 8px;'>
                    <div style='font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px;'>Model Confidence</div>
                    <div style='font-size: 2.6rem; font-weight: 900; color: {verdict_color}; line-height: 1;'>
                        {conf_pct}%
                    </div>
                    <div style='font-size: 0.82rem; margin-top: 4px; opacity: 0.85;'>
                        Authentic: {pred_res['probabilities']['authentic']*100:.1f}% | Tampered: {pred_res['probabilities']['tampered']*100:.1f}%
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Forensic Vital Signs Pill Row (6 quick diagnostic triage badges)
    v1, v2, v3, v4, v5, v6 = st.columns(6)
    with v1:
        st.markdown(f"<div class='forensic-pill'><b>🔒 Hash:</b>&nbsp;SHA-256 Valid</div>", unsafe_allow_html=True)
    with v2:
        meta_status = "🟡 Software Flag" if meta["editing_software_detected"] else ("🟢 Hardware EXIF" if meta["has_exif"] else "⚪ Stripped EXIF")
        st.markdown(f"<div class='forensic-pill'><b>🏷️ EXIF:</b>&nbsp;{meta_status}</div>", unsafe_allow_html=True)
    with v3:
        ela_status = "⚠️ Suspicious" if ela_stats["suspicious_ela"] else "🟢 Uniform"
        st.markdown(f"<div class='forensic-pill'><b>🔍 ELA:</b>&nbsp;{ela_status}</div>", unsafe_allow_html=True)
    with v4:
        noise_status = "⚠️ Inconsistent" if noise_stats["suspicious_noise"] else "🟢 Homogeneous"
        st.markdown(f"<div class='forensic-pill'><b>📡 Noise:</b>&nbsp;{noise_status}</div>", unsafe_allow_html=True)
    with v5:
        comb_status = "⚠️ Comb Artifact" if hist_stats["comb_effect_detected"] else "🟢 Smooth Curve"
        st.markdown(f"<div class='forensic-pill'><b>📈 Hist:</b>&nbsp;{comb_status}</div>", unsafe_allow_html=True)
    with v6:
        cm_status = f"🚨 {cm_res['inliers_count']} Inliers" if cm_res["evidence_found"] else "🟢 No Clones"
        st.markdown(f"<div class='forensic-pill'><b>✂️ Clones:</b>&nbsp;{cm_status}</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 0.8rem;'></div>", unsafe_allow_html=True)


    # ==========================================
    # SYSTEMATIC TAB NAVIGATION (8 FOCUSED TABS)
    # ==========================================
    tabs = st.tabs([
        "📊 Overview & Integrity",
        "🏷️ EXIF Metadata",
        "🔍 Error Level (ELA)",
        "📡 Noise Residuals",
        "📈 Histogram Analysis",
        "✂️ Copy-Move Detection",
        "🤖 ML & 24-D Features",
        "📑 Case Report & Export",
    ])


    # ------------------------------------------
    # TAB 1: OVERVIEW & INTEGRITY
    # ------------------------------------------
    with tabs[0]:
        st.markdown("#### 📊 Evidence Ingestion & Cryptographic Chain of Custody")
        col_img, col_meta = st.columns([1.1, 1])

        with col_img:
            st.image(pil_img, caption=f"Original Evidence: {props['filename']}", width="stretch")

        with col_meta:
            with st.container(border=True):
                st.markdown("##### Cryptographic Hash (SHA-256)")
                st.code(props['sha256'], language="text")
                st.caption(
                    "Cryptographic hash of exact raw bytes preserves chain of custody and "
                    "guarantees that evidence was not altered during forensic ingestion."
                )

            with st.container(border=True):
                st.markdown("##### Physical & Format Specifications")
                spec_df = pd.DataFrame([
                    {"Property": "File Name", "Value": props["filename"]},
                    {"Property": "File Format", "Value": props["format"]},
                    {"Property": "Image Dimensions", "Value": props["dimensions"]},
                    {"Property": "Resolution", "Value": f"{props['megapixels']} Megapixels"},
                    {"Property": "Color Space", "Value": f"{props['color_mode']} ({props['channels']} channels)"},
                    {"Property": "File Size (KB)", "Value": f"{props['file_size_kb']} KB"},
                    {"Property": "File Size (Bytes)", "Value": f"{props['file_size_bytes']} bytes"},
                ])
                st.dataframe(spec_df, hide_index=True, width="stretch")


    # ------------------------------------------
    # TAB 2: EXIF & METADATA FORENSICS
    # ------------------------------------------
    with tabs[1]:
        st.markdown("#### 🏷️ Structural Metadata & EXIF Hardware Signatures")
        
        st.info(f"**Forensic Finding:** {meta['forensic_summary']}")

        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            with st.container(border=True):
                st.markdown("##### Camera Hardware & Timestamps")
                meta_df1 = pd.DataFrame([
                    {"Metadata Tag": "Camera Make", "Extracted Value": meta["camera_make"]},
                    {"Metadata Tag": "Camera Model", "Extracted Value": meta["camera_model"]},
                    {"Metadata Tag": "Capture Timestamp", "Extracted Value": meta["date_time"]},
                    {"Metadata Tag": "Software Signature", "Extracted Value": meta["software"]},
                    {"Metadata Tag": "Image Orientation", "Extracted Value": meta["orientation"]},
                    {"Metadata Tag": "Extraction Engine", "Extracted Value": meta["source"]},
                ])
                st.dataframe(meta_df1, hide_index=True, width="stretch")

        with col_meta2:
            with st.container(border=True):
                st.markdown("##### Optical & Exposure Parameters")
                meta_df2 = pd.DataFrame([
                    {"Metadata Tag": "ISO Sensitivity", "Extracted Value": meta["iso"]},
                    {"Metadata Tag": "Exposure Time", "Extracted Value": meta["exposure_time"]},
                    {"Metadata Tag": "Focal Length", "Extracted Value": meta["focal_length"]},
                    {"Metadata Tag": "Recorded Dimensions", "Extracted Value": meta["dimensions"]},
                    {"Metadata Tag": "Has EXIF Tags", "Extracted Value": "Yes" if meta["has_exif"] else "No"},
                    {"Metadata Tag": "Software Flag", "Extracted Value": "Detected" if meta["editing_software_detected"] else "None"},
                ])
                st.dataframe(meta_df2, hide_index=True, width="stretch")

        if meta.get("gps"):
            with st.container(border=True):
                st.markdown("##### 🛰️ Geolocation (GPS Coordinates)")
                st.json(meta["gps"])

        if meta.get("raw_tags"):
            with st.expander(f"Inspect All Raw Metadata Tags ({len(meta['raw_tags'])} tags parsed)"):
                st.json(meta["raw_tags"])


    # ------------------------------------------
    # TAB 3: ERROR LEVEL ANALYSIS (ELA)
    # ------------------------------------------
    with tabs[2]:
        st.markdown("#### 🔍 Error Level Analysis (JPEG Quantization Error)")
        ela_data = full_res["ela"]
        ela_img = ela_data["image"]

        # Side-by-side plot
        fig_ela = plot_ela_comparison(pil_img, ela_img, ela_stats)
        st.pyplot(fig_ela, width="stretch")
        plt.close(fig_ela)

        # Metrics row
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            st.metric("Mean Error", f"{ela_stats['mean_error']:.2f}")
        with e2:
            st.metric("Peak Error (Max)", f"{ela_stats['max_error']:.2f}")
        with e3:
            st.metric("95th Percentile", f"{ela_stats['percentile_95']:.2f}")
        with e4:
            st.metric("Block Discrepancy", f"{ela_stats['block_discrepancy_ratio']:.2f}x")

        with st.container(border=True):
            st.markdown(f"**Forensic Assessment**: {ela_stats['interpretation']}")
            st.caption(f"ℹ️ {ela_stats['disclaimer']}")

        with st.expander("📚 Technical Theory: How Error Level Analysis Works"):
            st.markdown(
                """
                **JPEG Compression Forensics Principle:**
                When an image is compressed as a JPEG, each 8×8 pixel block undergoes Discrete Cosine Transform (DCT)
                and is divided by a quantization table. In a natural, unmodified JPEG, the error between the original
                and re-compressed image reaches an equilibrium state across all blocks.
                
                When an object is spliced into an image or modified:
                - The spliced region introduces high-frequency DCT discrepancies.
                - When re-compressed at a controlled quality factor (e.g. 90%), modified areas compress at different rates,
                  appearing as bright, highlighted clusters in the amplified difference map.
                """
            )


    # ------------------------------------------
    # TAB 4: NOISE RESIDUAL ANALYSIS
    # ------------------------------------------
    with tabs[3]:
        st.markdown("#### 📡 High-Frequency Sensor Noise & Local Variance Inconsistency")
        noise_data = full_res["noise"]
        noise_vis = noise_data["vis_image"]
        local_vars = noise_data["local_vars"]

        fig_noise = plot_noise_map(pil_img, noise_vis, local_vars)
        st.pyplot(fig_noise, width="stretch")
        plt.close(fig_noise)

        n1, n2, n3, n4 = st.columns(4)
        with n1:
            st.metric("Global Noise Variance", f"{noise_stats['global_noise_variance']:.2f}")
        with n2:
            st.metric("Signal-to-Noise Ratio", f"{noise_stats['snr_db']:.2f} dB")
        with n3:
            st.metric("Residual Kurtosis", f"{noise_stats['residual_kurtosis']:.2f}")
        with n4:
            st.metric("Inconsistent Blocks", f"{noise_stats['inconsistency_ratio']*100:.1f}%")

        with st.container(border=True):
            st.markdown(f"**Forensic Assessment**: {noise_stats['interpretation']}")

        with st.expander("📚 Technical Theory: Photo-Response Non-Uniformity (PRNU) & Noise Residuals"):
            st.markdown(
                """
                **Sensor Noise Profile:**
                Every digital camera sensor (CCD/CMOS) leaves an imperceptible high-frequency physical fingerprint
                known as Photo-Response Non-Uniformity (PRNU), caused by microscopic imperfections in individual silicon pixels.
                
                - High-pass median filtering separates structural content from the camera's high-frequency noise floor.
                - In an authentic single-capture photograph, noise variance is homogeneous across the frame.
                - Spliced content from a different camera or generated synthetically exhibits a mismatched noise profile,
                  producing sharp local variance divergence.
                """
            )


    # ------------------------------------------
    # TAB 5: HISTOGRAM ANALYSIS
    # ------------------------------------------
    with tabs[4]:
        st.markdown("#### 📈 Histogram Distribution & Non-Linear Artifacts")
        
        fig_hist = plot_histogram(hist_stats)
        st.pyplot(fig_hist, width="stretch")
        plt.close(fig_hist)

        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.metric("Luminance Mean", f"{hist_stats['gray_mean']:.1f}")
        with h2:
            st.metric("Shannon Entropy", f"{hist_stats['gray_entropy']:.2f} bits")
        with h3:
            st.metric("Comb Effect Detected", "YES" if hist_stats["comb_effect_detected"] else "NO")
        with h4:
            st.metric("Empty Bins in Span", f"{hist_stats['empty_bins_in_range']}")

        with st.container(border=True):
            st.markdown(f"**Histogram Assessment**: {hist_stats['comb_summary']}")

        with st.expander("📚 Technical Theory: Non-Linear Histogram Artifacts & Comb Effect"):
            st.markdown(
                """
                **Comb Effect & Histogram Equalization:**
                When an image undergoes global contrast stretching, levels adjustment, or histogram equalization:
                - Pixel intensities are mapped non-linearly across the [0, 255] range.
                - Because discrete integers cannot be split into fractions, periodic zero-count bins ('comb gaps') appear.
                - Shannon entropy measures the information density; natural uncompressed photographs typically measure 7.0–7.8 bits.
                """
            )


    # ------------------------------------------
    # TAB 6: COPY-MOVE FORGERY DETECTION
    # ------------------------------------------
    with tabs[5]:
        st.markdown("#### ✂️ Copy-Move Duplicate Region Forgery Detection")
        cm_data = full_res["copy_move"]
        cm_vis = cm_data["annotated_image"]

        fig_cm = plot_copy_move_matches(cm_vis, cm_res)
        st.pyplot(fig_cm, width="stretch")
        plt.close(fig_cm)

        cm1, cm2, cm3 = st.columns(3)
        with cm1:
            st.metric("Detector Algorithm", cm_res["algorithm"])
        with cm2:
            st.metric("Extracted Keypoints", cm_res["keypoints_count"])
        with cm3:
            st.metric("RANSAC Inlier Pairs", cm_res["inliers_count"])

        if cm_res["evidence_found"]:
            st.error(f"🚨 **Copy-Move Alert:** {cm_res['summary']}")
        else:
            st.success(f"✅ **Copy-Move Analysis:** {cm_res['summary']}")

        st.caption(f"Details: {cm_res['details']}")

        with st.expander("📚 Technical Theory: SIFT Matching & RANSAC Geometric Verification"):
            st.markdown(
                """
                **Intra-Image Feature Matching Pipeline:**
                1. **Scale-Invariant Feature Transform (SIFT)** extracts scale- and rotation-invariant gradient descriptors.
                2. Descriptors are matched against themselves across the image using k-Nearest Neighbors (k-NN).
                3. **Lowe's Ratio Test** filters ambiguous descriptors by comparing the closest distance to the second-closest.
                4. **Spatial Euclidean Filtering** removes trivial self-matches and adjacent neighboring points (< 35px).
                5. **Random Sample Consensus (RANSAC)** verifies whether matched keypoint pairs share a coherent affine transformation matrix,
                   ruling out random texture similarities.
                """
            )


    # ------------------------------------------
    # TAB 7: MACHINE LEARNING & 24-D FEATURES
    # ------------------------------------------
    with tabs[6]:
        st.markdown("#### 🤖 Supervised Classifier & 24-Dimensional Forensic Vector")
        
        # Class probability bars
        with st.container(border=True):
            st.markdown("##### Calibrated Prediction Probabilities")
            p_c1, p_c2 = st.columns(2)
            with p_c1:
                st.write(f"**Authentic**: `{pred_res['probabilities']['authentic'] * 100:.2f}%`")
                st.progress(float(pred_res['probabilities']['authentic']))
            with p_c2:
                st.write(f"**Tampered / Manipulated**: `{pred_res['probabilities']['tampered'] * 100:.2f}%`")
                st.progress(float(pred_res['probabilities']['tampered']))

        # Top influential features for this prediction
        st.markdown("##### Top Forensic Features Influencing This Decision")
        top_df = pd.DataFrame(pred_res["top_features"])
        st.dataframe(top_df, hide_index=True, width="stretch")

        # Full 24-Dimensional Feature Table
        with st.expander("Inspect Full 24-Dimensional Forensic Feature Vector"):
            feat_dict = pred_res["features"]
            feat_rows = []
            for idx, fname in enumerate(FEATURE_NAMES):
                feat_rows.append({
                    "Index": idx,
                    "Feature Name": fname,
                    "Extracted Value": round(feat_dict.get(fname, 0.0), 4),
                    "Forensic Significance": FEATURE_DESCRIPTIONS.get(fname, ""),
                })
            st.dataframe(pd.DataFrame(feat_rows), hide_index=True, width="stretch")

        # Validated Model Benchmark Metrics (Held-out test set)
        if pred_res.get("metrics"):
            st.markdown("##### Validated Model Benchmark Metrics (Held-Out Test Set)")
            m = pred_res["metrics"]
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Test Accuracy", f"{m['accuracy']*100:.2f}%")
            with m2:
                st.metric("Precision", f"{m['precision']*100:.2f}%")
            with m3:
                st.metric("Recall", f"{m['recall']*100:.2f}%")
            with m4:
                st.metric("F1-Score", f"{m['f1_score']:.3f}")

            if "confusion_matrix" in m:
                fig_cm_eval = plot_confusion_matrix(m["confusion_matrix"])
                st.pyplot(fig_cm_eval, width="content")
                plt.close(fig_cm_eval)


    # ------------------------------------------
    # TAB 8: CONSOLIDATED REPORT & EXPORT
    # ------------------------------------------
    with tabs[7]:
        st.markdown("#### 📑 Consolidated Forensic Evidence Dossier & Reports")

        with st.container(border=True):
            st.markdown("##### Multi-Technique Forensic Evidence Matrix")
            evidence_summary_data = [
                {
                    "Forensic Technique": "Cryptographic Hash",
                    "Observed Metric": f"SHA-256: {props['sha256'][:16]}...",
                    "Diagnostic Finding": "Unmodified raw evidence bytes validated.",
                    "Tampering Indication": "Neutral",
                },
                {
                    "Forensic Technique": "EXIF / Metadata",
                    "Observed Metric": f"Software: {meta['software']} | Make: {meta['camera_make']}",
                    "Diagnostic Finding": meta["status"],
                    "Tampering Indication": "Low" if not meta["editing_software_detected"] else "Medium (Re-save flag)",
                },
                {
                    "Forensic Technique": "Error Level Analysis",
                    "Observed Metric": f"Mean: {ela_stats['mean_error']:.2f} | Block Ratio: {ela_stats['block_discrepancy_ratio']:.2f}x",
                    "Diagnostic Finding": ela_stats["interpretation"],
                    "Tampering Indication": "High" if ela_stats["suspicious_ela"] else "Low",
                },
                {
                    "Forensic Technique": "Sensor Noise Residual",
                    "Observed Metric": f"SNR: {noise_stats['snr_db']:.2f} dB | Inconsistency: {noise_stats['inconsistency_ratio']*100:.1f}%",
                    "Diagnostic Finding": noise_stats["interpretation"],
                    "Tampering Indication": "High" if noise_stats["suspicious_noise"] else "Low",
                },
                {
                    "Forensic Technique": "Histogram & Non-Linear",
                    "Observed Metric": f"Entropy: {hist_stats['gray_entropy']:.2f} bits | Comb: {hist_stats['comb_effect_detected']}",
                    "Diagnostic Finding": hist_stats["comb_summary"],
                    "Tampering Indication": "Medium" if hist_stats["comb_effect_detected"] else "Low",
                },
                {
                    "Forensic Technique": "Copy-Move Cloning",
                    "Observed Metric": f"RANSAC Inliers: {cm_res['inliers_count']} | Keypoints: {cm_res['keypoints_count']}",
                    "Diagnostic Finding": cm_res["summary"],
                    "Tampering Indication": "Definitive" if cm_res["evidence_found"] else "Low",
                },
                {
                    "Forensic Technique": "Machine Learning (RF)",
                    "Observed Metric": f"Confidence: {pred_res['confidence_percent']}%",
                    "Diagnostic Finding": f"Predicted {pred_label}",
                    "Tampering Indication": "High" if is_tamp else "Low",
                },
            ]
            st.dataframe(pd.DataFrame(evidence_summary_data), hide_index=True, width="stretch")

        # Download Buttons
        st.markdown("##### Download Official Forensic Examination Report")
        pdf_bytes = report_bundle["pdf_bytes"]
        md_text = report_bundle["markdown"]
        case_id = report_bundle["case_id"]

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.download_button(
                label="📄 Download Official PDF Forensic Report",
                data=pdf_bytes,
                file_name=f"Forensic_Report_{case_id}.pdf",
                mime="application/pdf",
                type="primary",
                width="stretch",
            )
        with r_col2:
            st.download_button(
                label="📝 Download Markdown / Text Dossier",
                data=md_text,
                file_name=f"Forensic_Report_{case_id}.md",
                mime="text/markdown",
                width="stretch",
            )

        # Statutory Examiner Disclaimer
        st.warning(
            "⚠️ **MANDATORY EXAMINER DISCLAIMER:**\n\n"
            '"This is an automated forensic assessment. The result is not conclusive proof of image manipulation '
            'and should be interpreted by a qualified forensic examiner."'
        )
