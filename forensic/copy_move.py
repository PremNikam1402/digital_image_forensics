"""Copy-Move Forgery Detection Module.

Implements feature-based duplicate region detection using SIFT / ORB,
k-NN descriptor matching, Lowe's ratio test, spatial distance filtering,
and RANSAC geometric verification.
Never fabricates matches: reports 'No significant copy-move evidence detected'
if no statistically verified duplicate region exists.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, Union, List, Optional
from PIL import Image
import numpy as np
import cv2


def detect_copy_move(
    image: Union[Image.Image, np.ndarray],
    algorithm: str = "SIFT",
    ratio_threshold: float = 0.75,
    min_spatial_distance: float = 35.0,
    min_inliers: int = 5
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Detect copy-move cloning within an image.
    
    Parameters:
        image: PIL Image or RGB/BGR numpy array
        algorithm: 'SIFT' (recommended for precision) or 'ORB'
        ratio_threshold: Lowe's ratio test threshold (default 0.75)
        min_spatial_distance: Minimum pixel distance between matched keypoints
                              to avoid trivial self/neighbor matches (default 35.0 px)
        min_inliers: Minimum geometrically verified keypoint pairs to flag cloning (default 5)
        
    Returns:
        Tuple of:
          - annotated_image: RGB image with verified matching vectors drawn
          - results: Detailed dictionary of copy-move findings
    """
    if isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        img_rgb = image.copy()
        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Step 1: Initialize Feature Detector
    use_sift = (algorithm.upper() == "SIFT")
    detector = None

    if use_sift:
        try:
            detector = cv2.SIFT_create(nfeatures=2000)
        except Exception:
            use_sift = False

    if not use_sift or detector is None:
        algorithm = "ORB"
        detector = cv2.ORB_create(nfeatures=2000)

    # Step 2: Extract Keypoints and Descriptors
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    annotated = img_rgb.copy()

    if keypoints is None or descriptors is None or len(keypoints) < 10:
        return annotated, {
            "evidence_found": False,
            "algorithm": algorithm,
            "keypoints_count": len(keypoints) if keypoints else 0,
            "matched_pairs_count": 0,
            "inliers_count": 0,
            "summary": "No significant copy-move evidence detected.",
            "details": "Insufficient distinctive feature keypoints detected in the image.",
            "pairs": [],
        }

    # Step 3: Feature Matching (Intra-image matching)
    # Match descriptors against themselves using BFMatcher
    if use_sift:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Query k=6 nearest neighbors for each descriptor to allow finding genuine duplicate regions
    # while skipping immediate self-matches and spatial neighbors
    try:
        knn_matches = bf.knnMatch(descriptors, descriptors, k=min(8, len(descriptors)))
    except Exception:
        knn_matches = []

    candidate_pairs: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
    seen_pairs = set()

    pts1 = []
    pts2 = []

    for match_list in knn_matches:
        if len(match_list) < 2:
            continue

        q_idx = match_list[0].queryIdx
        p1 = keypoints[q_idx].pt

        # Find best match that satisfies minimum spatial distance
        best_match = None
        second_match = None

        for m in match_list:
            t_idx = m.trainIdx
            if t_idx == q_idx:
                continue  # skip exact self-match
            
            p2 = keypoints[t_idx].pt
            dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])

            if dist >= min_spatial_distance:
                if best_match is None:
                    best_match = m
                elif second_match is None:
                    second_match = m
                    break

        if best_match is not None and second_match is not None:
            # Lowe's ratio test between best spatial candidate and second candidate
            if best_match.distance < ratio_threshold * second_match.distance:
                idx_a, idx_b = min(q_idx, best_match.trainIdx), max(q_idx, best_match.trainIdx)
                if (idx_a, idx_b) not in seen_pairs:
                    seen_pairs.add((idx_a, idx_b))
                    pt_a = keypoints[idx_a].pt
                    pt_b = keypoints[idx_b].pt
                    candidate_pairs.append((pt_a, pt_b, best_match.distance))
                    pts1.append(pt_a)
                    pts2.append(pt_b)

    # Step 4: RANSAC Geometric Consistency Verification
    verified_pairs = []
    inliers_count = 0

    if len(pts1) >= 4:
        np_pts1 = np.float32(pts1)
        np_pts2 = np.float32(pts2)

        # Estimate affine transformation with RANSAC to verify consistent geometric translation
        transform, inlier_mask = cv2.estimateAffinePartial2D(
            np_pts1, np_pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=2000,
            confidence=0.99
        )

        if inlier_mask is not None:
            inliers = inlier_mask.ravel().astype(bool)
            inliers_count = int(np.sum(inliers))
            
            for idx, is_inlier in enumerate(inliers):
                if is_inlier:
                    verified_pairs.append(candidate_pairs[idx])

    # Step 5: Visualization & Final Assessment
    evidence_found = inliers_count >= min_inliers

    if evidence_found:
        # Draw verified duplicate regions with distinct cyan/red keypoints and connecting lines
        for pt_a, pt_b, _ in verified_pairs:
            p1_int = (int(round(pt_a[0])), int(round(pt_a[1])))
            p2_int = (int(round(pt_b[0])), int(round(pt_b[1])))
            
            # Keypoints
            cv2.circle(annotated, p1_int, 5, (255, 60, 60), -1)    # Red source point
            cv2.circle(annotated, p2_int, 5, (0, 220, 255), -1)    # Cyan target point
            # Connecting line
            cv2.line(annotated, p1_int, p2_int, (50, 255, 50), 2)  # Green vector
            
        summary = (
            f"Suspicious copy-move forgery detected: {inliers_count} geometrically consistent "
            f"cloned keypoint pairs identified via {algorithm} with RANSAC verification."
        )
    else:
        # No reliable duplicate region found
        summary = "No significant copy-move evidence detected."

    return annotated, {
        "evidence_found": evidence_found,
        "algorithm": algorithm,
        "keypoints_count": len(keypoints),
        "candidate_matches_count": len(candidate_pairs),
        "inliers_count": inliers_count,
        "summary": summary,
        "details": (
            f"Analyzed {len(keypoints)} keypoints. {len(candidate_pairs)} candidate pairs passed Lowe's test; "
            f"{inliers_count} pairs were geometrically verified by RANSAC."
        ),
        "pairs": [
            {
                "pt1": [round(p[0][0], 1), round(p[0][1], 1)],
                "pt2": [round(p[1][0], 1), round(p[1][1], 1)],
                "distance": round(p[2], 2)
            }
            for p in verified_pairs
        ],
    }
