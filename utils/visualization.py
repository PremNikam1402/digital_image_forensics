"""Visualization Helpers for Digital Image Forensics Dashboard.

Generates Matplotlib and OpenCV figures for histograms, ELA side-by-side,
noise residual maps, local variance heatmaps, and copy-move keypoint overlays.
"""

from __future__ import annotations

from typing import Dict, Any, Union, Optional
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure


# Configure clean styling for college project demonstration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_histogram(hist_data: Dict[str, Any]) -> matplotlib.figure.Figure:
    """Plot dual-panel Grayscale and RGB multi-channel histograms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=100)

    # Panel 1: Grayscale Histogram
    gray_hist = np.array(hist_data["hist_gray"])
    ax1.plot(gray_hist, color="black", lw=1.8, label="Luminance (Gray)")
    ax1.fill_between(range(256), gray_hist, color="gray", alpha=0.3)
    ax1.set_title("Grayscale Luminance Histogram", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Pixel Intensity (0-255)", fontsize=9)
    ax1.set_ylabel("Pixel Frequency", fontsize=9)
    ax1.set_xlim([0, 255])
    ax1.legend(loc="upper right", fontsize=8)

    # Annotate stats
    ax1.text(
        0.03, 0.95,
        f"Mean: {hist_data.get('gray_mean', 0.0):.1f}\n"
        f"Std: {hist_data.get('gray_std', 0.0):.1f}\n"
        f"Entropy: {hist_data.get('gray_entropy', 0.0):.2f} bits",
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#ccc")
    )

    # Panel 2: RGB Channel Histograms
    r_hist = np.array(hist_data["hist_r"])
    g_hist = np.array(hist_data["hist_g"])
    b_hist = np.array(hist_data["hist_b"])

    ax2.plot(r_hist, color="#d9534f", lw=1.5, label="Red Channel")
    ax2.plot(g_hist, color="#5cb85c", lw=1.5, label="Green Channel")
    ax2.plot(b_hist, color="#0275d8", lw=1.5, label="Blue Channel")
    ax2.set_title("RGB Multi-Channel Color Histograms", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Pixel Intensity (0-255)", fontsize=9)
    ax2.set_ylabel("Pixel Frequency", fontsize=9)
    ax2.set_xlim([0, 255])
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


def plot_ela_comparison(
    orig_image: Image.Image,
    ela_image: Image.Image,
    ela_stats: Dict[str, Any]
) -> matplotlib.figure.Figure:
    """Plot side-by-side comparison of original image and enhanced ELA."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=100)

    ax1.imshow(orig_image)
    ax1.set_title("Original Evidence Image", fontsize=11, fontweight="bold")
    ax1.axis("off")

    ax2.imshow(ela_image)
    quality = ela_stats.get("quality_level", 90)
    scale = ela_stats.get("scale_factor", 15)
    ax2.set_title(f"Error Level Analysis (JPEG Q={quality}, {scale}x Amplified)", fontsize=11, fontweight="bold")
    ax2.axis("off")

    fig.tight_layout()
    return fig


def plot_noise_map(
    orig_image: Image.Image,
    noise_vis: np.ndarray,
    local_vars: np.ndarray
) -> matplotlib.figure.Figure:
    """Plot three-panel image: original, noise residual, and local variance heatmap."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.2), dpi=100)

    ax1.imshow(orig_image)
    ax1.set_title("Original Image", fontsize=10, fontweight="bold")
    ax1.axis("off")

    ax2.imshow(noise_vis, cmap="gray")
    ax2.set_title("Sensor Noise Residual Map", fontsize=10, fontweight="bold")
    ax2.axis("off")

    # Local variance heatmap
    im3 = ax3.imshow(local_vars, cmap="magma", interpolation="nearest")
    ax3.set_title("Local Noise Variance Map", fontsize=10, fontweight="bold")
    ax3.axis("off")
    cbar = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    return fig


def plot_copy_move_matches(
    annotated_image: np.ndarray,
    results: Dict[str, Any]
) -> matplotlib.figure.Figure:
    """Plot copy-move duplicate region visualization with matching vectors."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=100)

    ax.imshow(annotated_image)
    inliers = results.get("inliers_count", 0)
    algo = results.get("algorithm", "SIFT")
    
    if results.get("evidence_found", False):
        title = f"Copy-Move Forgery Detected: {inliers} Verified Cloned Keypoints ({algo})"
        color = "darkred"
    else:
        title = f"Copy-Move Analysis ({algo}): No Suspicious Duplicate Regions"
        color = "darkgreen"

    ax.set_title(title, fontsize=11, fontweight="bold", color=color)
    ax.axis("off")

    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm_dict: Dict[str, Any]) -> matplotlib.figure.Figure:
    """Plot Confusion Matrix heatmap for test set evaluation."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=100)
    
    raw_cm = np.array(cm_dict.get("raw_matrix", [[0, 0], [0, 0]]))
    im = ax.imshow(raw_cm, cmap="Blues", interpolation="nearest")

    classes = ["Authentic", "Tampered"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=10, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("True Class", fontsize=10)
    ax.set_title("Random Forest Test Confusion Matrix", fontsize=11, fontweight="bold")

    # Annotate counts inside cells
    thresh = raw_cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = raw_cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=14, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
