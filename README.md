# Digital Image Forensics and Tampering Detection System

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![UI Framework](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest%20(Scikit--Learn)-green.svg)](https://scikit-learn.org/)
[![Computer Vision](https://img.shields.io/badge/CV-OpenCV%20%2B%20Pillow-orange.svg)](https://opencv.org/)

A complete, production-grade **Bachelor of Engineering (BE) Final Year Project** implementing an automated multi-technique digital image forensics and forgery detection system.

The system combines classical signal and physical image forensic techniques (Error Level Analysis, Sensor Noise Discrepancy, Histogram Entropy & Comb Artifacts, SIFT/ORB Copy-Move Detection) with a supervised Random Forest classifier trained on a 24-dimensional unified forensic feature vector.

---

## 📌 Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement & Objectives](#problem-statement--objectives)
3. [System Architecture](#system-architecture)
4. [Forensic Techniques Implemented](#forensic-techniques-implemented)
5. [24-Dimensional Feature Vector](#24-dimensional-feature-vector)
6. [Machine Learning Pipeline](#machine-learning-pipeline)
7. [Installation & Setup](#installation--setup)
8. [Running the Application](#running-the-application)
9. [Dataset Setup & Model Training](#dataset-setup--model-training)
10. [ExifTool Setup (Optional)](#exiftool-setup-optional)
11. [Forensic Report Generation](#forensic-report-generation)
12. [BE Viva Voce Preparation Guide (Q&A)](#be-viva-voce-preparation-guide-qa)
13. [Limitations & Future Scope](#limitations--future-scope)

---

## 1. Project Overview

In the modern digital era, photo-editing tools and generative techniques make image tampering trivial. Forensic investigators, law enforcement, and journalism platforms require scientifically sound, multi-faceted methods to detect tampering.

This system analyzes an uploaded image across **six orthogonal forensic dimensions**:
1. **Chain of Custody & Cryptographic Integrity**: SHA-256 byte hashing and structural format verification.
2. **Metadata & EXIF Forensics**: Hardware signatures (Make, Model, Lens), timestamps, GPS coordinates, and editing software indicators (Photoshop, GIMP, Canva, etc.) using ExifTool with Pillow fallback.
3. **Error Level Analysis (ELA)**: Re-compression difference analysis to detect spliced objects saved at differing compression rates.
4. **Sensor Noise Analysis**: High-frequency residual extraction via median filtering and block-level local variance mapping to identify spliced regions from cameras with different ISO/sensors.
5. **Histogram & Non-Linear Artifacts**: Color channel moments, Shannon entropy, and comb-like pattern detection caused by contrast stretching and histogram equalization.
6. **Copy-Move Forgery Detection**: Intra-image SIFT/ORB keypoint matching, Lowe's ratio test, spatial Euclidean distance filtering, and RANSAC geometric affine verification to identify duplicated regions.

---

## 2. Problem Statement & Objectives

### Problem Statement
Most existing image tampering detection tools either rely solely on a single heuristic (such as ELA, which is prone to false positives on sharp edges) or operate as an opaque "black-box" deep learning model without explainable evidence. 

### Objectives
- Build a modular, transparent, and non-fabricated forensic workbench.
- Implement classical algorithms without artificial or random outputs.
- Build a supervised Machine Learning pipeline that fuses statistical and signal features into a calibrated prediction.
- Provide a court-ready, downloadable PDF examination report with full chain of custody and statutory examiner disclaimers.

---

## 3. System Architecture

```
User Upload (JPG / PNG / TIFF / BMP)
   │
   ▼
[ 1. Image Preprocessing & Integrity ] ───> Computes SHA-256 Hash & Validates Pixels
   │
   ├──────────────────────────┬──────────────────────────┬──────────────────────────┐
   ▼                          ▼                          ▼                          ▼
[ 2. Metadata Engine ]   [ 3. ELA Engine ]      [ 4. Noise Engine ]   [ 5. Histogram Engine ]
 (ExifTool / Pillow)     (Recompression Q=90)   (Median Residual)      (Entropy & Comb Test)
   │                          │                          │                          │
   └──────────────────────────┼──────────────────────────┴──────────────────────────┘
                              ▼
                 [ 6. Copy-Move Engine ]
                 (SIFT/ORB + RANSAC Clustering)
                              │
                              ▼
                 [ 7. Feature Extractor ] ───> Extracts 24 Numerical Metrics
                              │
                              ▼
             [ 8. Random Forest Classifier ]
                   (Supervised Model)
                              │
                              ▼
             [ 9. Streamlit Dashboard & Report ]
             (Tabs A-K, Side-by-Side, PDF Export)
```

---

## 4. Forensic Techniques Implemented

### 4.1 Cryptographic Integrity & File Validation
- Computes raw byte SHA-256 hash before loading pixels, ensuring unalterable chain of custody.
- Verifies image stream integrity with Pillow `verify()` to block corrupted or malformed payloads.

### 4.2 EXIF & Metadata Forensics
- Uses `exiftool` CLI if present on PATH for comprehensive EXIF, XMP, and IPTC extraction.
- Seamlessly falls back to native Pillow `_getexif()` and `ExifTags` if ExifTool is not installed.
- Detects editing software tags (e.g., Photoshop, Canva, GIMP) without jumping to false conclusions: clearly differentiates between ordinary format re-saving and content tampering.

### 4.3 Error Level Analysis (ELA)
- **Mathematical Foundation**: Re-compresses the image at a controlled JPEG quality factor $Q$ (default 90). The absolute pixel difference is calculated:
  $$\Delta(x,y) = |I_{\text{original}}(x,y) - I_{Q}(x,y)|$$
- Multiplied by an enhancement factor ($15\times$) for visual inspection.
- Computes mean error, max error, 95th percentile, and spatial block discrepancy ratio.
- In authentic images, degradation is uniform across similar surfaces. In spliced images, newly inserted regions display noticeably higher or lower error energy.

### 4.4 Sensor Noise Residual & Local Variance
- **Principle**: Every camera sensor produces unique high-frequency sensor pattern noise (PRNU). Splicing an object from another camera introduces an incompatible noise floor.
- Extracts noise residual via median filtering: $R(x, y) = I(x, y) - \text{median}_{3\times3}(I(x, y))$.
- Calculates global noise variance, Signal-to-Noise Ratio (SNR in dB), residual kurtosis, and block-level ($16\times16$) local variance anomalies.

### 4.5 Histogram Moments & Shannon Entropy
- Computes 256-bin grayscale and R, G, B histograms.
- **Shannon Entropy**: Measures information richness:
  $$H = -\sum_{i=0}^{255} p_i \log_2(p_i)$$
- **Comb Effect Detection**: Identifies periodic empty bins in the histogram caused by non-linear contrast stretching or histogram equalization.

### 4.6 Copy-Move Forgery Detection
- Extracts SIFT scale-invariant features (with ORB fallback).
- Performs intra-image k-NN descriptor matching ($k=8$).
- Eliminates trivial self-matches and adjacent texture points via Euclidean spatial distance filtering ($>35$ px).
- Applies **Lowe's Ratio Test** ($d_1 / d_2 < 0.75$).
- Performs **RANSAC Affine Geometric Verification** on candidate matching vectors to eliminate false-positive texture pairs.
- If no consistent cloned region exists, strictly outputs: *"No significant copy-move evidence detected."*

---

## 5. 24-Dimensional Feature Vector

The feature extractor (`forensic/feature_extraction.py`) produces a standardized 24-element numeric vector:

| Index | Feature Name | Description |
|:---:|:---|:---|
| 0 | `ela_mean_error` | Average pixel difference after JPEG re-compression at Q=90 |
| 1 | `ela_std_error` | Standard deviation of ELA error |
| 2 | `ela_max_error` | Peak ELA pixel difference observed |
| 3 | `ela_p95_error` | 95th percentile ELA error (isolates high-contrast spliced boundaries) |
| 4 | `ela_block_variance` | Spatial variance across 16x16 ELA blocks |
| 5 | `ela_block_ratio` | Max-to-min block error ratio (spatial inconsistency metric) |
| 6 | `noise_global_variance` | Variance of median-filter noise residual |
| 7 | `noise_global_std` | Standard deviation of noise residual |
| 8 | `noise_snr_db` | Estimated Signal-to-Noise Ratio in decibels |
| 9 | `noise_kurtosis` | Kurtosis of noise residual (non-Gaussian sensor signature) |
| 10 | `noise_skewness` | Skewness of noise residual distribution |
| 11 | `noise_local_var_std` | Standard deviation of local block noise variances |
| 12 | `noise_inconsistency_ratio` | Fraction of blocks deviating >2.5 sigma from global noise |
| 13 | `gray_mean` | Global average luminance intensity [0-255] |
| 14 | `gray_std` | Global luminance standard deviation |
| 15 | `gray_skewness` | Third standardized moment of luminance histogram |
| 16 | `gray_kurtosis` | Fourth standardized moment of luminance histogram |
| 17 | `gray_entropy` | Shannon entropy of grayscale histogram (bits) |
| 18 | `rgb_entropy_mean` | Average Shannon entropy across R, G, B channels |
| 19 | `hist_comb_ratio` | Ratio of empty bins in active histogram range (comb effect) |
| 20 | `cm_keypoints_count` | Total SIFT/ORB keypoints detected |
| 21 | `cm_inliers_count` | RANSAC-verified duplicate keypoint pairs (copy-move) |
| 22 | `meta_has_exif` | Binary flag (1 if EXIF metadata present, 0 if missing) |
| 23 | `meta_software_flag` | Binary flag (1 if editing software signature detected) |

---

## 6. Machine Learning Pipeline

- **Classifier**: `RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)`
- **Data Splitting**: Stratified 80/20 train/test split.
- **Data Leakage Prevention**: Feature scaling and thresholds computed strictly on training data; evaluation conducted only on unseen test sets.
- **Class Balancing**: Balanced sample weighting to handle imbalanced authentic vs tampered sets.
- **Genuine Metrics Evaluated**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - Confusion Matrix (TN, FP, FN, TP)
- Serialized to `models/random_forest_forensic.joblib` and metrics saved to `models/model_metrics.json`.

---

## 7. Installation & Setup

### Prerequisites
- Python 3.11, 3.12, or 3.13 installed.

### Step-by-Step Installation

#### 1. Open Terminal or PowerShell
Navigate to the project root directory:
```bash
cd "digital forensic of images"
```

#### 2. Create and Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)*

On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 8. Running the Application

Launch the Streamlit dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 9. Dataset Setup & Model Training

The project comes with a verified sample dataset generator in `ml/dataset.py` so you can train and evaluate the model immediately out-of-the-box!

### Training Out-of-the-Box:
```bash
python ml/train.py
```
This will:
1. Generate realistic authentic and tampered sample images in `data/sample/`.
2. Extract the 24-dimensional feature matrix.
3. Perform a stratified 80/20 train/test split.
4. Train the Random Forest classifier.
5. Save the model to `models/random_forest_forensic.joblib` and metrics to `models/model_metrics.json`.

### Using Public Benchmark Datasets (Optional):
The training pipeline supports public image tampering datasets:
- **CASIA Image Tampering Dataset (v1.0 & v2.0)**: Place authentic images in `data/raw/authentic` and spliced/tampered images in `data/raw/tampered`.
- **Columbia Image Splicing Dataset**: Place pristine in `data/raw/authentic` and spliced in `data/raw/tampered`.
- **CoMoFoD / COVERAGE Copy-Move Datasets**: Place originals in `data/raw/authentic` and forged in `data/raw/tampered`.

Run training on your custom dataset directory:
```bash
python ml/train.py --data-dir data/raw
```

---

## 10. ExifTool Setup (Optional)

The application automatically uses **Pillow's native EXIF parser** as a fallback. Installing ExifTool gives access to deep XMP and IPTC tags:

1. Download **ExifTool for Windows** from [https://exiftool.org/](https://exiftool.org/).
2. Extract `exiftool(-k).exe` and rename it to `exiftool.exe`.
3. Move `exiftool.exe` to a folder in your PATH (e.g., `C:\Windows\` or any folder added to your Environment Variables).
4. Verify by running `exiftool -ver` in your terminal.

---

## 11. Forensic Report Generation

The application generates an official, downloadable **PDF Forensic Report** using ReportLab.
Each report includes:
- Case Analysis ID (UUID)
- Examination Timestamp
- Evidence Cryptographic SHA-256 Hash
- Image Resolution, Dimensions, and Format
- Detailed EXIF Findings & Hardware Tags
- Multi-Technique Evidence Matrix (ELA, Noise, Histogram, Copy-Move)
- Machine Learning Classification & Calibrated Confidence Score
- Mandatory Examiner Legal Disclaimer

---

## 12. BE Viva Voce Preparation Guide (Q&A)

### Q1: What is Error Level Analysis (ELA) and how does it detect manipulation?
**Answer**: ELA works on the principle that lossy JPEG compression degrades an image at a specific error rate across continuous surfaces. When an image is spliced by pasting an element from an external picture (or edited in software), that spliced region will have undergone a different number of compression cycles or used a different JPEG quantization table. When we re-save the entire image at a controlled quality factor (e.g., 90%) and compute the pixel delta $|I_{\text{orig}} - I_{\text{resaved}}|$, authentic regions display uniform error energy, whereas spliced regions exhibit higher or lower error energy.

### Q2: Why does ELA show bright outlines along sharp edges even in authentic photos?
**Answer**: High-frequency components (sharp edges, high-contrast boundaries) are naturally penalized more heavily by discrete cosine transform (DCT) quantization in the JPEG compression standard. Therefore, bright ELA edges are normal for natural textures. A qualified examiner looks for *asymmetric regional discrepancy*—a smooth surface displaying abnormally high error compared to adjacent smooth surfaces.

### Q3: Does the presence of "Adobe Photoshop" in EXIF mean an image is tampered?
**Answer**: No. An image can be opened in Photoshop simply to resize it, convert color spaces, or re-save it without modifying any pixel content. Hence, software tags are treated as contextual indicators and never as standalone proof of malicious tampering.

### Q4: How does your Copy-Move detection work and how do you prevent false matches?
**Answer**: We extract scale-invariant feature descriptors using SIFT (Scale-Invariant Feature Transform). We match descriptors within the image using k-NN and Lowe's ratio test ($d_1 / d_2 < 0.75$). To prevent false matches from uniform background textures or identical adjacent pixels:
1. We enforce a minimum Euclidean spatial distance ($>35$ pixels) between matched points.
2. We run RANSAC (Random Sample Consensus) to estimate an affine geometric transformation between matched pairs. Random false matches do not share a consistent transformation vector, so RANSAC discards them as outliers.

### Q5: How do you prevent data leakage during ML training?
**Answer**: We perform a stratified train/test split *before* any feature normalization or model fitting. All model decisions, class weight balancing, and performance evaluations are strictly measured on the isolated test set that the classifier never saw during training.

---

## 13. Limitations & Future Scope

### Current Limitations
- **Aggressive Social Media Compression**: Platforms like WhatsApp or Instagram compress images heavily and strip all EXIF metadata, reducing the fidelity of high-frequency noise analysis.
- **Deepfakes / AI Generation**: Diffusion and GAN-generated faces do not necessarily exhibit copy-move or traditional splicing artifacts.

### Future Scope
- **Deep Learning CNN Forensics**: Integrating MesoNet or EfficientNet for detecting AI-generated synthetic faces.
- **CFA / Demosaicing Artifact Analysis**: Detecting inconsistencies in the Color Filter Array interpolation patterns of camera Bayer filters.
- **Shadow & Lighting Consistency**: Geometric validation of light source angles across objects in the scene.

---

## 📜 Statutory Forensic Disclaimer
> *"This is an automated forensic assessment. The result is not conclusive proof of image manipulation and should be interpreted by a qualified forensic examiner."*
