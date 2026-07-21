"""
Holistic (texture/edge-based) feature extraction for crowd counting.

This follows the classical "holistic regression" approach to crowd counting
(e.g. Chan & Vasconcelos, 2008): instead of only counting individually
detected people (which fails under occlusion in dense crowds), extract
low-level image statistics that correlate with how many people are present
-- edge density, foreground texture, corner/blob counts, plus the raw
detection count from a pretrained person detector as one extra signal --
and let a regressor learn the mapping from these features to the true
count.
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

FEATURE_NAMES = [
    "edge_density",
    "edge_density_half",
    "foreground_ratio",
    "contour_count",
    "contour_count_small",
    "corner_count",
    "lbp_entropy",
    "gray_std",
    "perimeter_density",
    "hog_detections",
]


def _entropy(hist):
    hist = hist / (hist.sum() + 1e-8)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def extract_features(image_bgr, hog_detector=None, resize_width=384):
    """Compute a fixed-length feature vector from a BGR image.

    hog_detector: optional cv2.HOGDescriptor (or callable returning a count)
    used as one extra feature. If None, that feature is 0.
    """
    h, w = image_bgr.shape[:2]
    scale = resize_width / w
    img = cv2.resize(image_bgr, (resize_width, int(h * scale)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) Edge density (full res + half res -> captures density at two scales)
    edges = cv2.Canny(gray, 60, 150)
    edge_density = float(np.mean(edges > 0))

    small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))
    edges_half = cv2.Canny(small, 60, 150)
    edge_density_half = float(np.mean(edges_half > 0))

    # 2) Foreground ratio via adaptive threshold vs. estimated background
    blur = cv2.medianBlur(gray, 9)
    diff = cv2.absdiff(gray, blur)
    _, fg_mask = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
    foreground_ratio = float(np.mean(fg_mask > 0))

    # 3) Contours on the edge map -> proxy for distinct head/body blobs
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    contour_count = float(len(contours))
    contour_count_small = float(sum(1 for a in areas if 2 < a < 150))
    perimeter_density = float(sum(cv2.arcLength(c, True) for c in contours)) / (img.shape[0] * img.shape[1])

    # 4) Corner density (Shi-Tomasi) -> more texture/clutter with more people
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.05, minDistance=4)
    corner_count = float(0 if corners is None else len(corners))

    # 5) Local Binary Pattern texture entropy
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    lbp_entropy = _entropy(lbp_hist.astype(np.float64))

    gray_std = float(np.std(gray))

    # 6) Raw pretrained-detector count as an extra weak signal (saturates in dense crowds)
    hog_detections = 0.0
    if hog_detector is not None:
        try:
            boxes, _ = hog_detector.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.03)
            hog_detections = float(len(boxes))
        except Exception:
            hog_detections = 0.0

    feats = [
        edge_density,
        edge_density_half,
        foreground_ratio,
        contour_count,
        contour_count_small,
        corner_count,
        lbp_entropy,
        gray_std,
        perimeter_density,
        hog_detections,
    ]
    return np.array(feats, dtype=np.float64)
