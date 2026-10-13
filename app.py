"""Digital Image Forensics and Tampering Detection System - Streamlit Dashboard.

A complete multi-technique forensic analysis workbench featuring:
- Image Integrity & SHA-256 Hashing
- EXIF & Structural Metadata Forensics (ExifTool / Pillow)
- Error Level Analysis (ELA) with Interactive Quality Tuning
- High-Frequency Sensor Noise & Local Variance Inconsistency
- Histogram Moments, Shannon Entropy & Comb Anomaly Detection
- SIFT/ORB Copy-Move Duplicate Region Forgery Detection
- 24-Dimensional Unified Forensic Feature Extraction
- Supervised Random Forest Classification & Calibrated Probability
- Automated Professional PDF & Markdown Report Generation
"""

from __future__ import annotations

import io
import os
import json
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Forensic modules
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


# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Digital Image Forensics System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #10b981;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    .metric-card {
        background-color: rgba(128, 128, 128, 0.08);
        border-radius: 8px;
        padding: 14px;
        border-left: 4px solid #3b82f6;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 12px;
    }
    .verdict-box-auth {
        background-color: rgba(16, 185, 129, 0.12);
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        color: #10b981;
    }
    .verdict-box-tamp {
        background-color: rgba(239, 68, 68, 0.12);
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        color: #ef4444;
    }
    .disclaimer-banner {
        background-color: rgba(245, 158, 11, 0.12);
        border-left: 5px solid #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.88rem;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# INITIALIZATION & CACHED RESOURCES
# ==========================================
@st.cache_resource
def get_predictor() -> ForensicPredictor:
    """Load cached inference model."""
    return ForensicPredictor()


predictor = get_predictor()


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/search-property.png",
        width=64,
    )
    st.markdown("### Forensic Examination Controls")

    uploaded_file = st.file_uploader(
        "Upload Evidence Image",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"],
        help="Upload JPG, JPEG, PNG, or TIFF image to examine.",
    )

    st.markdown("---")
    st.markdown("#### Analysis Settings")
    
    ela_quality = st.slider(
        "ELA JPEG Quality",
        min_value=60,
        max_value=98,
        value=90,
        step=1,
        help="Compression quality for Error Level Analysis comparison (Standard: 90-95%)",
    )

    copy_move_algo = st.selectbox(
        "Copy-Move Feature Detector",
        options=["SIFT", "ORB"],
        index=0,
        help="SIFT provides superior scale & rotation invariance; ORB is binary and fast.",
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

    run_analysis_btn = st.button("🚀 Run Forensic Analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("#### Machine Learning Status")
    if predictor.model_loaded and predictor.metrics:
        acc = predictor.metrics.get("accuracy", 0.0) * 100
        f1 = predictor.metrics.get("f1_score", 0.0)
        st.success(f"Model: Random Forest\nTest Accuracy: {acc:.1f}%\nF1-Score: {f1:.3f}")
    elif predictor.model_loaded:
        st.success("Model: Random Forest (Loaded)")
    else:
        st.warning("Model: Not trained yet\n(Using conservative forensic rules)")


# ==========================================
# HEADER
# ==========================================
st.markdown("<div class='main-header'>DIGITAL IMAGE FORENSICS & TAMPERING DETECTION SYSTEM</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Multi-Technique Digital Forensic Examination Workbench • BE Project</div>",
    unsafe_allow_html=True,
)


# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
if uploaded_file is None:
    # Landing page / Instructions
    st.info("👋 **Welcome to the Digital Image Forensics System.** Please upload an evidence image from the left sidebar to begin automated forensic examination.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class='metric-card'>
            <h4>🔬 Forensic Techniques</h4>
            <ul>
                <li>Cryptographic SHA-256 Hashing</li>
                <li>EXIF / XMP Metadata Extraction</li>
                <li>Error Level Analysis (ELA)</li>
                <li>High-Frequency Sensor Noise Analysis</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class='metric-card'>
            <h4>📐 Structural & Signal Metrics</h4>
            <ul>
                <li>Shannon Entropy & Kurtosis</li>
                <li>Histogram Comb Artifact Detection</li>
                <li>SIFT / ORB Copy-Move Cloned Keypoints</li>
                <li>RANSAC Affine Consistency Verification</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class='metric-card'>
            <h4>🤖 Machine Learning & Reporting</h4>
            <ul>
                <li>Supervised Random Forest Classifier</li>
                <li>Calibrated Probabilistic Confidence</li>
                <li>Non-Fabricated Objective Results</li>
                <li>Downloadable PDF Examination Reports</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Display sample images from data/sample if available
    sample_dir = Path("data/sample")
    if sample_dir.exists():
        auth_samples = list(sample_dir.glob("authentic/*.jpg"))
        tamp_samples = list(sample_dir.glob("tampered/*.jpg"))
        if auth_samples or tamp_samples:
            st.markdown("---")
            st.markdown("### 📁 Quick Test with Included Verified Samples")
            st.write("You can test the system with bundled test samples located in `data/sample/`.")
            s_col1, s_col2 = st.columns(2)
            if auth_samples:
                with s_col1:
                    st.caption(f"Authentic Sample: `{auth_samples[0].name}`")
                    st.image(str(auth_samples[0]), width=320)
            if tamp_samples:
                with s_col2:
                    st.caption(f"Tampered Sample: `{tamp_samples[0].name}`")
                    st.image(str(tamp_samples[0]), width=320)

else:
    # Save uploaded file bytes to temporary file for exact EXIF and byte hashing
    raw_bytes = uploaded_file.getvalue()
    temp_dir = tempfile.mkdtemp()
    temp_image_path = Path(temp_dir) / uploaded_file.name
    with open(temp_image_path, "wb") as f:
        f.write(raw_bytes)

    # Step 1: Validation
    is_valid, val_msg, pil_img = validate_image(temp_image_path)
    if not is_valid or pil_img is None:
        st.error(f"❌ Image Validation Failed: {val_msg}")
        st.stop()

    # Step 2: Properties
    props = get_image_properties(pil_img, raw_bytes=raw_bytes, filename=uploaded_file.name)

    # Perform or retrieve cached analysis
    analysis_key = f"forensic_{props['sha256']}_{ela_quality}_{copy_move_algo}_{lowe_ratio}"
    
    if analysis_key not in st.session_state or run_analysis_btn:
        with st.spinner("Executing multi-technique digital forensic examination..."):
            # Run prediction and full forensic suite
            pred_results = predictor.predict(
                image=pil_img,
                image_path=temp_image_path,
                copy_move_algo=copy_move_algo,
                ela_quality=ela_quality,
            )
            # Compile report
            report_bundle = generate_forensic_report(pred_results, props)
            
            st.session_state[analysis_key] = {
                "results": pred_results,
                "report": report_bundle,
            }

    data = st.session_state[analysis_key]
    pred_res = data["results"]
    report_bundle = data["report"]
    full_res = pred_res["full_results"]

    # ==========================================
    # FORENSIC NAVIGATION TABS (A through K)
    # ==========================================
    tabs = st.tabs([
        "A. Overview",
        "B. Metadata / EXIF",
        "C. ELA Analysis",
        "D. Noise Analysis",
        "E. Histogram",
        "F. Copy-Move",
        "G. Extracted Features",
        "H. ML Prediction",
        "I. Evidence Summary",
        "J. Final Verdict",
        "K. Download Report",
    ])

    # ------------------------------------------
    # TAB A: IMAGE OVERVIEW & INTEGRITY
    # ------------------------------------------
    with tabs[0]:
        st.markdown("### A. Image Overview & Chain of Custody")
        col_img, col_info = st.columns([1.1, 1])

        with col_img:
            st.image(pil_img, caption=f"Evidence: {props['filename']}", use_container_width=True)

        with col_info:
            st.markdown("#### Cryptographic & Structural Integrity")
            st.code(f"SHA-256: {props['sha256']}", language="text")
            
            p_df = pd.DataFrame([
                {"Property": "File Name", "Value": props["filename"]},
                {"Property": "Format", "Value": props["format"]},
                {"Property": "Dimensions", "Value": props["dimensions"]},
                {"Property": "Resolution", "Value": f"{props['megapixels']} Megapixels"},
                {"Property": "Color Mode", "Value": f"{props['color_mode']} ({props['channels']} channels)"},
                {"Property": "File Size (KB)", "Value": f"{props['file_size_kb']} KB"},
                {"Property": "File Size (Bytes)", "Value": f"{props['file_size_bytes']} bytes"},
            ])
            st.dataframe(p_df, hide_index=True, use_container_width=True)

    # ------------------------------------------
    # TAB B: EXIF / METADATA FORENSICS
    # ------------------------------------------
    with tabs[1]:
        st.markdown("### B. Metadata & Structural EXIF Forensics")
        meta = full_res["metadata"]
        
        status_color = "🟢" if meta["has_exif"] and not meta["editing_software_detected"] else ("🟡" if meta["editing_software_detected"] else "⚪")
        st.markdown(f"**Metadata Status**: {status_color} **{meta['status']}** (Extracted via `{meta['source']}`)")
        st.info(meta["forensic_summary"])

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### Hardware & Capture Settings")
            m_df1 = pd.DataFrame([
                {"Tag": "Camera Make", "Value": meta["camera_make"]},
                {"Tag": "Camera Model", "Value": meta["camera_model"]},
                {"Tag": "Date/Time Original", "Value": meta["date_time"]},
                {"Tag": "Software Signature", "Value": meta["software"]},
                {"Tag": "Orientation", "Value": meta["orientation"]},
            ])
            st.dataframe(m_df1, hide_index=True, use_container_width=True)

        with col_m2:
            st.markdown("#### Exposure & Optical Settings")
            m_df2 = pd.DataFrame([
                {"Tag": "ISO Speed", "Value": meta["iso"]},
                {"Tag": "Exposure Time", "Value": meta["exposure_time"]},
                {"Tag": "Focal Length", "Value": meta["focal_length"]},
                {"Tag": "Image Dimensions", "Value": meta["dimensions"]},
            ])
            st.dataframe(m_df2, hide_index=True, use_container_width=True)

        if meta.get("gps"):
            st.markdown("#### 🛰️ Geolocation (GPS)")
            st.json(meta["gps"])

        if meta.get("raw_tags"):
            with st.expander(f"Inspect All Raw Metadata Tags ({len(meta['raw_tags'])} tags)"):
                st.json(meta["raw_tags"])

    # ------------------------------------------
    # TAB C: ERROR LEVEL ANALYSIS (ELA)
    # ------------------------------------------
    with tabs[2]:
        st.markdown("### C. Error Level Analysis (ELA)")
        ela_data = full_res["ela"]
        ela_img = ela_data["image"]
        ela_stats = ela_data["stats"]

        st.markdown(f"**Re-compression Quality**: `{ela_stats['quality_level']}%` | **Enhancement Amplification**: `{ela_stats['scale_factor']}x`")
        
        fig_ela = plot_ela_comparison(pil_img, ela_img, ela_stats)
        st.pyplot(fig_ela, use_container_width=True)
        plt.close(fig_ela)

        # Statistics Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Error", f"{ela_stats['mean_error']:.2f}")
        c2.metric("Max Error", f"{ela_stats['max_error']:.2f}")
        c3.metric("95th Percentile", f"{ela_stats['percentile_95']:.2f}")
        c4.metric("Block Discrepancy", f"{ela_stats['block_discrepancy_ratio']:.2f}x")

        st.markdown(f"**Interpretation**: {ela_stats['interpretation']}")
        st.caption(f"ℹ️ {ela_stats['disclaimer']}")

    # ------------------------------------------
    # TAB D: NOISE ANALYSIS
    # ------------------------------------------
    with tabs[3]:
        st.markdown("### D. High-Frequency Noise & Local Variance Inconsistency")
        noise_data = full_res["noise"]
        noise_vis = noise_data["vis_image"]
        local_vars = noise_data["local_vars"]
        noise_stats = noise_data["stats"]

        fig_noise = plot_noise_map(pil_img, noise_vis, local_vars)
        st.pyplot(fig_noise, use_container_width=True)
        plt.close(fig_noise)

        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Global Noise Variance", f"{noise_stats['global_noise_variance']:.2f}")
        n2.metric("Signal-to-Noise Ratio (SNR)", f"{noise_stats['snr_db']:.2f} dB")
        n3.metric("Residual Kurtosis", f"{noise_stats['residual_kurtosis']:.2f}")
        n4.metric("Inconsistent Blocks", f"{noise_stats['inconsistency_ratio']*100:.1f}%")

        st.markdown(f"**Interpretation**: {noise_stats['interpretation']}")

    # ------------------------------------------
    # TAB E: HISTOGRAM ANALYSIS
    # ------------------------------------------
    with tabs[4]:
        st.markdown("### E. Histogram Distribution & Non-Linear Artifacts")
        hist_data = full_res["histogram"]

        fig_hist = plot_histogram(hist_data)
        st.pyplot(fig_hist, use_container_width=True)
        plt.close(fig_hist)

        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Luminance Mean", f"{hist_data['gray_mean']:.1f}")
        h1.metric("Luminance Std", f"{hist_data['gray_std']:.1f}")
        h2.metric("Shannon Entropy", f"{hist_data['gray_entropy']:.2f} bits")
        h2.metric("Luminance Skewness", f"{hist_data['gray_skewness']:.2f}")
        h3.metric("Comb Effect Detected", "YES" if hist_data["comb_effect_detected"] else "NO")
        h4.metric("Empty Bins in Range", f"{hist_data['empty_bins_in_range']}")

        st.markdown(f"**Histogram Assessment**: {hist_data['comb_summary']}")

    # ------------------------------------------
    # TAB F: COPY-MOVE FORGERY DETECTION
    # ------------------------------------------
    with tabs[5]:
        st.markdown("### F. Copy-Move Duplicate Region Forgery Detection")
        cm_data = full_res["copy_move"]
        cm_vis = cm_data["annotated_image"]
        cm_res = cm_data["results"]

        fig_cm = plot_copy_move_matches(cm_vis, cm_res)
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("Algorithm", cm_res["algorithm"])
        cm2.metric("Extracted Keypoints", cm_res["keypoints_count"])
        cm3.metric("RANSAC Inlier Pairs", cm_res["inliers_count"])

        if cm_res["evidence_found"]:
            st.error(f"🚨 {cm_res['summary']}")
        else:
            st.success(f"✅ {cm_res['summary']}")
            
        st.caption(f"Details: {cm_res['details']}")

    # ------------------------------------------
    # TAB G: EXTRACTED FEATURES
    # ------------------------------------------
    with tabs[6]:
        st.markdown("### G. Unified 24-Dimensional Forensic Feature Vector")
        st.write("This standardized numerical feature vector integrates physical, statistical, and signal indicators into a machine-learning-ready representation:")
        
        feat_dict = pred_res["features"]
        feat_rows = []
        for idx, fname in enumerate(FEATURE_NAMES):
            feat_rows.append({
                "Index": idx,
                "Feature Name": fname,
                "Extracted Value": round(feat_dict.get(fname, 0.0), 4),
                "Description": FEATURE_DESCRIPTIONS.get(fname, ""),
            })
        st.dataframe(pd.DataFrame(feat_rows), hide_index=True, use_container_width=True)

    # ------------------------------------------
    # TAB H: MACHINE LEARNING PREDICTION
    # ------------------------------------------
    with tabs[7]:
        st.markdown("### H. Supervised Machine Learning Classifier")
        is_tamp = pred_res["is_tampered"]
        pred_label = pred_res["prediction"]
        conf_pct = pred_res["confidence_percent"]

        if is_tamp:
            st.markdown(
                f"""
                <div class='verdict-box-tamp'>
                    <h2>🚨 ML Classification: {pred_label}</h2>
                    <h4>Calculated Model Confidence: {conf_pct}%</h4>
                    <p>Engine: {pred_res['source']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class='verdict-box-auth'>
                    <h2>✅ ML Classification: {pred_label}</h2>
                    <h4>Calculated Model Confidence: {conf_pct}%</h4>
                    <p>Engine: {pred_res['source']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Class Probabilities")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.write(f"**Authentic**: `{pred_res['probabilities']['authentic'] * 100:.2f}%`")
            st.progress(float(pred_res['probabilities']['authentic']))
        with p_col2:
            st.write(f"**Tampered / Manipulated**: `{pred_res['probabilities']['tampered'] * 100:.2f}%`")
            st.progress(float(pred_res['probabilities']['tampered']))

        # Feature Importance / Contributing Features
        st.markdown("#### Most Influential Forensic Features for This Prediction")
        top_df = pd.DataFrame(pred_res["top_features"])
        st.dataframe(top_df, hide_index=True, use_container_width=True)

        # Show training test metrics if available
        if pred_res.get("metrics"):
            st.markdown("---")
            st.markdown("#### Validated Model Benchmark Metrics (Held-Out Test Set)")
            m = pred_res["metrics"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
            m2.metric("Precision", f"{m['precision']*100:.2f}%")
            m3.metric("Recall", f"{m['recall']*100:.2f}%")
            m4.metric("F1-Score", f"{m['f1_score']:.3f}")

            if "confusion_matrix" in m:
                fig_cm_eval = plot_confusion_matrix(m["confusion_matrix"])
                st.pyplot(fig_cm_eval, use_container_width=False)
                plt.close(fig_cm_eval)

    # ------------------------------------------
    # TAB I: EVIDENCE SUMMARY
    # ------------------------------------------
    with tabs[8]:
        st.markdown("### I. Consolidated Forensic Evidence Matrix")
        st.write("Cross-technique comparison separating signal heuristics from supervised classification:")

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
                "Observed Metric": f"Mean: {ela_stats['mean_error']} | Block Ratio: {ela_stats['block_discrepancy_ratio']}x",
                "Diagnostic Finding": ela_stats["interpretation"],
                "Tampering Indication": "High" if ela_stats["suspicious_ela"] else "Low",
            },
            {
                "Forensic Technique": "Sensor Noise Residual",
                "Observed Metric": f"SNR: {noise_stats['snr_db']} dB | Inconsistency: {noise_stats['inconsistency_ratio']*100:.1f}%",
                "Diagnostic Finding": noise_stats["interpretation"],
                "Tampering Indication": "High" if noise_stats["suspicious_noise"] else "Low",
            },
            {
                "Forensic Technique": "Histogram & Non-Linear",
                "Observed Metric": f"Entropy: {hist_data['gray_entropy']} bits | Comb: {hist_data['comb_effect_detected']}",
                "Diagnostic Finding": hist_data["comb_summary"],
                "Tampering Indication": "Medium" if hist_data["comb_effect_detected"] else "Low",
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
        st.dataframe(pd.DataFrame(evidence_summary_data), hide_index=True, use_container_width=True)

    # ------------------------------------------
    # TAB J: FINAL VERDICT
    # ------------------------------------------
    with tabs[9]:
        st.markdown("### J. Final Forensic Examination Verdict")

        # Dynamic high-contrast banner matching current theme
        if is_tamp:
            st.error(
                f"### 🚨 FORENSIC ANALYSIS RESULT: {pred_label}\n\n"
                f"**Assessment**: The evidence exhibits statistical and signal anomalies consistent with image manipulation."
            )
        else:
            st.success(
                f"### ✅ FORENSIC ANALYSIS RESULT: {pred_label}\n\n"
                f"**Assessment**: The evidence exhibits uniform characteristics consistent with an authentic photograph."
            )

        # Main Forensic Dossier Card (Native container - adaptively dark/light mode aware)
        with st.container(border=True):
            st.markdown("#### 📋 Forensic Assessment Overview")
            
            c_v1, c_v2, c_v3 = st.columns(3)
            with c_v1:
                st.metric("Final Classification", pred_label)
            with c_v2:
                st.metric("ML Model Confidence", f"{conf_pct}%")
            with c_v3:
                engine_name = "Random Forest" if pred_res["model_loaded"] else "Forensic Rule Integrator"
                st.metric("Inference Engine", engine_name)

            st.divider()

            # Section 1: Supervised Machine Learning
            st.markdown("#### 🤖 1. Machine Learning Prediction")
            st.markdown(
                f"- **Prediction Verdict**: **{pred_label}**\n"
                f"- **Confidence Score**: **{conf_pct}%**\n"
                f"- **Class Probability Distribution**: Authentic = `{pred_res['probabilities']['authentic'] * 100:.2f}%` | Manipulated = `{pred_res['probabilities']['tampered'] * 100:.2f}%`\n"
                f"- **Model Architecture**: {pred_res['source']}"
            )

            st.divider()

            # Section 2: Traditional Forensic Evidence (Physical & Signal Analysis)
            st.markdown("#### 🔬 2. Traditional Forensic Evidence (Physical & Signal Analysis)")
            st.markdown(
                f"- **Error Level Analysis (ELA)**: {ela_stats['interpretation']}\n"
                f"- **Sensor Noise Residual**: {noise_stats['interpretation']}\n"
                f"- **Copy-Move Cloning Detection**: {cm_res['summary']}\n"
                f"- **Metadata & Structural EXIF**: {meta['forensic_summary']}\n"
                f"- **Histogram & Non-Linear Artifacts**: {full_res['histogram']['comb_summary']}"
            )

        # Statutory Forensic Disclaimer
        st.warning(
            "⚠️ **MANDATORY EXAMINER DISCLAIMER:**\n\n"
            '"This is an automated forensic assessment. The result is not conclusive proof of image manipulation '
            'and should be interpreted by a qualified forensic examiner."'
        )


    # ------------------------------------------
    # TAB K: DOWNLOAD FORENSIC REPORT
    # ------------------------------------------
    with tabs[10]:
        st.markdown("### K. Download Official Forensic Examination Report")
        st.write("Generate and download a certified forensic dossier for academic documentation or case filing:")

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
                use_container_width=True,
            )
        with r_col2:
            st.download_button(
                label="📝 Download Markdown / Text Report",
                data=md_text,
                file_name=f"Forensic_Report_{case_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with st.expander("Preview Markdown Examination Dossier"):
            st.markdown(f"```markdown\n{md_text}\n```")
