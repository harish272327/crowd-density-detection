"""
Loads the trained crowd-counting model (ml/train_model.py output) and
exposes a simple predict_count() / classify_density() API for app.py.
"""
import json
import os

import joblib
import numpy as np

from ml.features import extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "crowd_count_model.joblib")
THRESHOLDS_PATH = os.path.join(MODELS_DIR, "density_thresholds.json")
REPORT_PATH = os.path.join(MODELS_DIR, "training_report.json")

_model = None
_thresholds = {"low_max": 10, "medium_max": 30}  # sane fallback if artifacts are missing
_report = {}

if os.path.exists(MODEL_PATH):
    _model = joblib.load(MODEL_PATH)
if os.path.exists(THRESHOLDS_PATH):
    with open(THRESHOLDS_PATH) as f:
        _thresholds = json.load(f)
if os.path.exists(REPORT_PATH):
    with open(REPORT_PATH) as f:
        _report = json.load(f)

MODEL_LOADED = _model is not None
TRAINING_NOTE = _report.get("note", "")
TEST_MAE = _report.get("test_mae")


def predict_count(image_bgr, hog_detector=None):
    """Predict the number of people in an image using the trained regressor.
    Falls back to None if no model artifact was found."""
    if _model is None:
        return None
    feats = extract_features(image_bgr, hog_detector=hog_detector).reshape(1, -1)
    pred = float(_model.predict(feats)[0])
    return max(0, round(pred))


def classify_density(count):
    """Bucket a predicted count into Low / Medium / High using the
    data-derived thresholds saved at training time."""
    low_max = _thresholds["low_max"]
    med_max = _thresholds["medium_max"]

    if count <= low_max:
        level = "Low"
        suggestion = "Crowd is sparse. No action needed."
    elif count <= med_max:
        level = "Medium"
        suggestion = "Moderate crowd. Keep monitoring the area."
    else:
        level = "High"
        suggestion = "Crowd is dense. Consider crowd-control measures or limiting further entry."
    return level, suggestion


def thresholds():
    return dict(_thresholds)
