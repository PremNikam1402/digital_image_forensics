"""Automated Forensic Examination Report Generator.

Generates complete, professional forensic examination reports in PDF and Markdown/Text formats.
Includes Case ID, SHA-256 hash, EXIF metadata, multi-technique findings,
ML prediction confidence, evidence summary, and statutory forensic disclaimers.
"""

from __future__ import annotations

import io
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)


DISCLAIMER_TEXT = (
    "DISCLAIMER: This is an automated forensic assessment. The result is not conclusive proof of "
    "image manipulation and should be interpreted by a qualified forensic examiner. Computational "
    "heuristics and machine learning models are diagnostic aids and must be corroborated by "
    "contextual evidence in accordance with digital forensics standards."
)


def generate_markdown_report(report_data: Dict[str, Any]) -> str:
    """Generate human-readable Markdown/Text forensic dossier."""
    props = report_data.get("properties", {})
    meta = report_data.get("metadata", {})
    ela = report_data.get("ela", {}).get("stats", {})
    noise = report_data.get("noise", {}).get("stats", {})
    hist = report_data.get("histogram", {})
    cm = report_data.get("copy_move", {}).get("results", {})
    ml = report_data.get("ml_prediction", {})

    lines = [
        "# DIGITAL IMAGE FORENSICS EXAMINATION REPORT",
        "=" * 60,
        f"**Case Analysis ID**: {report_data.get('case_id', 'N/A')}",
        f"**Examination Timestamp**: {report_data.get('timestamp', 'N/A')}",
        f"**Investigator / System**: Digital Forensic Tampering Detection System (v1.0)",
        "",
        "## 1. EVIDENCE INTEGRITY & CUSTODY",
        "-" * 40,
        f"- **File Name**: {props.get('filename', 'Unknown')}",
        f"- **Cryptographic SHA-256 Hash**: `{props.get('sha256', 'N/A')}`",
        f"- **Dimensions**: {props.get('dimensions', 'N/A')} ({props.get('megapixels', 'N/A')} MP)",
        f"- **Color Mode / Channels**: {props.get('color_mode', 'N/A')} ({props.get('channels', 'N/A')} channels)",
        f"- **File Size**: {props.get('file_size_kb', 0)} KB ({props.get('file_size_bytes', 0)} bytes)",
        "",
        "## 2. METADATA & EXIF FORENSICS",
        "-" * 40,
        f"- **Status**: {meta.get('status', 'Metadata unavailable')}",
        f"- **Extraction Engine**: {meta.get('source', 'N/A')}",
        f"- **Camera Make & Model**: {meta.get('camera_make', 'N/A')} {meta.get('camera_model', 'N/A')}",
        f"- **Original Timestamp**: {meta.get('date_time', 'N/A')}",
        f"- **Software Signature**: {meta.get('software', 'N/A')}",
        f"- **ISO / Exposure / Focal Length**: ISO {meta.get('iso', 'N/A')} | {meta.get('exposure_time', 'N/A')} | {meta.get('focal_length', 'N/A')}",
        f"- **GPS Coordinates**: {meta.get('gps', {}).get('coordinates', 'No GPS tags present') if meta.get('gps') else 'No GPS tags present'}",
        f"- **Summary**: {meta.get('forensic_summary', 'N/A')}",
        "",
        "## 3. ERROR LEVEL ANALYSIS (ELA)",
        "-" * 40,
        f"- **JPEG Re-compression Quality**: {ela.get('quality_level', 90)}%",
        f"- **Mean Error Level**: {ela.get('mean_error', 0.0)}",
        f"- **Max Error Level**: {ela.get('max_error', 0.0)}",
        f"- **95th Percentile Error**: {ela.get('percentile_95', 0.0)}",
        f"- **Block Discrepancy Ratio**: {ela.get('block_discrepancy_ratio', 1.0)}x",
        f"- **Assessment**: {ela.get('interpretation', 'N/A')}",
        "",
        "## 4. NOISE RESIDUAL & LOCAL VARIANCE",
        "-" * 40,
        f"- **Global Noise Variance**: {noise.get('global_noise_variance', 0.0)}",
        f"- **Signal-to-Noise Ratio (SNR)**: {noise.get('snr_db', 0.0)} dB",
        f"- **Residual Kurtosis**: {noise.get('residual_kurtosis', 0.0)}",
        f"- **Inconsistent Block Ratio**: {noise.get('inconsistency_ratio', 0.0) * 100:.2f}%",
        f"- **Assessment**: {noise.get('interpretation', 'N/A')}",
        "",
        "## 5. HISTOGRAM & NON-LINEAR ARTIFACTS",
        "-" * 40,
        f"- **Grayscale Mean / Std**: {hist.get('gray_mean', 0.0)} / {hist.get('gray_std', 0.0)}",
        f"- **Shannon Luminance Entropy**: {hist.get('gray_entropy', 0.0)} bits",
        f"- **Comb Artifact Detected**: {'YES (Non-linear manipulation detected)' if hist.get('comb_effect_detected') else 'NO (Smooth continuous distribution)'}",
        f"- **Assessment**: {hist.get('comb_summary', 'N/A')}",
        "",
        "## 6. COPY-MOVE CLONING DETECTION",
        "-" * 40,
        f"- **Algorithm Used**: {cm.get('algorithm', 'SIFT')}",
        f"- **Keypoints Extracted**: {cm.get('keypoints_count', 0)}",
        f"- **Candidate Pairs (Lowe's Test)**: {cm.get('candidate_matches_count', 0)}",
        f"- **RANSAC Verified Inliers**: {cm.get('inliers_count', 0)}",
        f"- **Verdict**: {cm.get('summary', 'No significant copy-move evidence detected.')}",
        "",
        "## 7. MACHINE LEARNING CLASSIFIER PREDICTION",
        "-" * 40,
        f"- **Model**: Random Forest Classifier (Supervised 24-Feature Vector)",
        f"- **Prediction**: **{ml.get('prediction', 'UNASSESSED')}**",
        f"- **ML Confidence**: {ml.get('confidence_percent', 0.0)}%",
        f"- **Class Probabilities**: Authentic={ml.get('probabilities', {}).get('authentic', 0.0):.4f} | Manipulated={ml.get('probabilities', {}).get('tampered', 0.0):.4f}",
        "",
        "## 8. CONSOLIDATED FORENSIC ASSESSMENT",
        "-" * 40,
        f"**FINAL VERDICT: {ml.get('prediction', 'LIKELY AUTHENTIC')}**",
        "",
        "## 9. LIMITATIONS & STATUTORY DISCLAIMER",
        "-" * 40,
        DISCLAIMER_TEXT,
        "=" * 60,
    ]
    return "\n".join(lines)


def generate_pdf_report(report_data: Dict[str, Any]) -> bytes:
    """Generate a clean, structured PDF report using ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1e3d59"),
        alignment=1,
        fontName="Helvetica-Bold"
    )
    
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#555555"),
        alignment=1
    )

    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#17252a"),
        spaceBefore=10,
        spaceAfter=4,
        fontName="Helvetica-Bold"
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#222222")
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#666666"),
        alignment=0
    )

    elements = []

    # Title & Header
    elements.append(Paragraph("DIGITAL IMAGE FORENSIC REPORT", title_style))
    elements.append(Paragraph("Automated Tampering Detection & Chain of Custody Analysis", subtitle_style))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3d59"), spaceAfter=8))

    # Case & Metadata Table
    props = report_data.get("properties", {})
    case_table_data = [
        [
            Paragraph("<b>Case Analysis ID:</b>", body_style),
            Paragraph(report_data.get("case_id", "N/A"), body_style),
            Paragraph("<b>Timestamp:</b>", body_style),
            Paragraph(report_data.get("timestamp", "N/A"), body_style),
        ],
        [
            Paragraph("<b>File Name:</b>", body_style),
            Paragraph(props.get("filename", "Unknown"), body_style),
            Paragraph("<b>Dimensions:</b>", body_style),
            Paragraph(f"{props.get('dimensions', 'N/A')} ({props.get('color_mode', 'N/A')})", body_style),
        ],
        [
            Paragraph("<b>SHA-256 Hash:</b>", body_style),
            Paragraph(f"<font size=6>{props.get('sha256', 'N/A')}</font>", body_style),
            Paragraph("<b>File Size:</b>", body_style),
            Paragraph(f"{props.get('file_size_kb', 0)} KB", body_style),
        ],
    ]

    t_case = Table(case_table_data, colWidths=[100, 170, 90, 180])
    t_case.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f9fa")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e0e0e0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(t_case)
    elements.append(Spacer(1, 10))

    # ML Verdict Banner
    ml = report_data.get("ml_prediction", {})
    is_tamp = ml.get("is_tampered", False)
    banner_bg = colors.HexColor("#ffebee") if is_tamp else colors.HexColor("#e8f5e9")
    banner_text_color = colors.HexColor("#c62828") if is_tamp else colors.HexColor("#2e7d32")
    
    verdict_p = Paragraph(
        f"<b>FINAL FORENSIC CLASSIFICATION: {ml.get('prediction', 'UNKNOWN')}</b><br/>"
        f"Calculated Model Confidence: <b>{ml.get('confidence_percent', 0.0)}%</b> "
        f"(Authentic Prob: {ml.get('probabilities', {}).get('authentic', 0.0):.4f} | "
        f"Tampered Prob: {ml.get('probabilities', {}).get('tampered', 0.0):.4f})",
        ParagraphStyle(
            "VerdictBanner",
            parent=styles["Normal"],
            fontSize=11,
            leading=15,
            textColor=banner_text_color,
            alignment=1
        )
    )
    t_verdict = Table([[verdict_p]], colWidths=[540])
    t_verdict.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), banner_bg),
        ("BOX", (0, 0), (-1, -1), 1, banner_text_color),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t_verdict)
    elements.append(Spacer(1, 10))

    # Forensic Findings Table
    elements.append(Paragraph("Multi-Technique Forensic Evidence Matrix", section_style))
    
    meta = report_data.get("metadata", {})
    ela = report_data.get("ela", {}).get("stats", {})
    noise = report_data.get("noise", {}).get("stats", {})
    hist = report_data.get("histogram", {})
    cm = report_data.get("copy_move", {}).get("results", {})

    evidence_rows = [
        [
            Paragraph("<b>Forensic Module</b>", body_style),
            Paragraph("<b>Primary Metrics</b>", body_style),
            Paragraph("<b>Diagnostic Finding</b>", body_style),
        ],
        [
            Paragraph("<b>EXIF / Metadata</b>", body_style),
            Paragraph(f"Source: {meta.get('source', 'None')}<br/>Camera: {meta.get('camera_make', 'N/A')}<br/>Software: {meta.get('software', 'None')}", body_style),
            Paragraph(meta.get("forensic_summary", "N/A"), body_style),
        ],
        [
            Paragraph("<b>Error Level (ELA)</b>", body_style),
            Paragraph(f"Mean: {ela.get('mean_error', 0.0)}<br/>Block Ratio: {ela.get('block_discrepancy_ratio', 1.0)}x<br/>P95: {ela.get('percentile_95', 0.0)}", body_style),
            Paragraph(ela.get("interpretation", "N/A"), body_style),
        ],
        [
            Paragraph("<b>Sensor Noise</b>", body_style),
            Paragraph(f"SNR: {noise.get('snr_db', 0.0)} dB<br/>Variance: {noise.get('global_noise_variance', 0.0)}<br/>Inconsistency: {noise.get('inconsistency_ratio', 0.0)*100:.1f}%", body_style),
            Paragraph(noise.get("interpretation", "N/A"), body_style),
        ],
        [
            Paragraph("<b>Histogram & Non-Linear</b>", body_style),
            Paragraph(f"Entropy: {hist.get('gray_entropy', 0.0):.2f} bits<br/>Comb Detected: {hist.get('comb_effect_detected', False)}", body_style),
            Paragraph(hist.get("comb_summary", "N/A"), body_style),
        ],
        [
            Paragraph("<b>Copy-Move Forgery</b>", body_style),
            Paragraph(f"Algorithm: {cm.get('algorithm', 'SIFT')}<br/>Candidate Pairs: {cm.get('candidate_matches_count', 0)}<br/>RANSAC Inliers: {cm.get('inliers_count', 0)}", body_style),
            Paragraph(cm.get("summary", "No significant copy-move evidence detected."), body_style),
        ],
    ]

    t_evidence = Table(evidence_rows, colWidths=[120, 160, 260])
    t_evidence.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f5")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(t_evidence)
    elements.append(Spacer(1, 12))

    # Top Features Summary Table
    top_feats = ml.get("top_features", [])
    if top_feats:
        elements.append(Paragraph("Top Forensic Feature Contributions", section_style))
        feat_rows = [
            [
                Paragraph("<b>Feature Metric</b>", body_style),
                Paragraph("<b>Extracted Value</b>", body_style),
                Paragraph("<b>Forensic Rationale</b>", body_style),
            ]
        ]
        for f in top_feats[:5]:
            feat_rows.append([
                Paragraph(f.get("name", ""), body_style),
                Paragraph(str(f.get("value", 0.0)), body_style),
                Paragraph(f.get("description", ""), body_style),
            ])
        t_feats = Table(feat_rows, colWidths=[140, 90, 310])
        t_feats.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f4f8")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e0e0e0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t_feats)
        elements.append(Spacer(1, 10))

    # Disclaimer Footer
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc"), spaceAfter=6))
    elements.append(Paragraph(f"<b>LEGAL & FORENSIC NOTICE:</b> {DISCLAIMER_TEXT}", disclaimer_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def generate_forensic_report(
    analysis_results: Dict[str, Any],
    image_properties: Dict[str, Any]
) -> Dict[str, Any]:
    """Compile analysis results into structured report with PDF and Markdown outputs."""
    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    report_dossier = {
        "case_id": case_id,
        "timestamp": timestamp_str,
        "properties": image_properties,
        "metadata": analysis_results["full_results"]["metadata"],
        "ela": analysis_results["full_results"]["ela"],
        "noise": analysis_results["full_results"]["noise"],
        "histogram": analysis_results["full_results"]["histogram"],
        "copy_move": analysis_results["full_results"]["copy_move"],
        "ml_prediction": analysis_results,
    }

    markdown_text = generate_markdown_report(report_dossier)
    pdf_bytes = generate_pdf_report(report_dossier)

    return {
        "case_id": case_id,
        "timestamp": timestamp_str,
        "markdown": markdown_text,
        "pdf_bytes": pdf_bytes,
        "dossier": report_dossier,
    }
