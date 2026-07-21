"""
Retrain the crowd-count model on REAL photos once you have some.

The synthetic-data model shipped with this app (models/crowd_count_model.joblib)
was trained on procedurally generated scenes because the sandbox that built
this app has no internet access to fetch a real dataset (ShanghaiTech, Mall,
UCF-CC-50, JHU-Crowd++, etc). The pipeline itself (features.py -> RandomForest)
is completely real and works the same on real photos -- you just need
(image, true_count) pairs to plug in.

--------------------------------------------------------------------
HOW TO USE
--------------------------------------------------------------------
1. Collect a folder of crowd photos and a CSV mapping filename -> people count:

     data/
       images/
         img001.jpg
         img002.jpg
         ...
       labels.csv        # columns: filename,count

   You can create labels.csv by hand-counting, or (much faster) by using a
   public dataset that ships point annotations -- sum the points per image
   to get `count`, e.g.:
     - Mall dataset (mall_dataset): ships a count per frame directly.
     - ShanghaiTech (Part A/B): .mat files with head-position points; count
       = number of points.
     - JHU-Crowd++ / UCF-QNRF: similar point-annotation format.

2. Run:
     python3 ml/train_on_real_data.py --images data/images --labels data/labels.csv

3. This overwrites models/crowd_count_model.joblib and
   models/density_thresholds.json in place -- restart the Flask app and it
   will pick up the new, real-data-trained model automatically.

No GPU or deep learning framework required -- same lightweight sklearn
pipeline as train_model.py, just fed real images instead of synthetic ones.
"""
import argparse
import csv
import json
import os

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import extract_features, FEATURE_NAMES

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(THIS_DIR), "models")


def load_real_dataset(images_dir, labels_csv):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    rows = []
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r["filename"], float(r["count"])))

    X, y = [], []
    for filename, count in rows:
        path = os.path.join(images_dir, filename)
        img = cv2.imread(path)
        if img is None:
            print(f"  [skip] could not read {path}")
            continue
        X.append(extract_features(img, hog_detector=hog))
        y.append(count)

    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder of real crowd photos")
    ap.add_argument("--labels", required=True, help="CSV with columns: filename,count")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    print(f"Loading real dataset from {args.images} / {args.labels} ...")
    X, y = load_real_dataset(args.images, args.labels)
    print(f"Loaded {len(y)} labeled images.")
    if len(y) < 20:
        print("WARNING: fewer than 20 labeled images. Model quality will be poor. "
              "Aim for at least a few hundred for a usable regressor.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_leaf=2,
                                      random_state=42, n_jobs=-1)),
    ])
    model.fit(X_train, y_train)

    if len(y_test) > 0:
        pred = np.clip(model.predict(X_test), 0, None)
        print(f"Test MAE: {mean_absolute_error(y_test, pred):.2f}  |  R2: {r2_score(y_test, pred):.3f}")

    q1, q2 = np.quantile(y, [1 / 3, 2 / 3])
    thresholds = {"low_max": float(round(q1)), "medium_max": float(round(q2))}
    print(f"New density thresholds -> Low: 0-{thresholds['low_max']:.0f}, "
          f"Medium: {thresholds['low_max']:.0f}-{thresholds['medium_max']:.0f}, "
          f"High: {thresholds['medium_max']:.0f}+")

    joblib.dump(model, os.path.join(MODELS_DIR, "crowd_count_model.joblib"))
    with open(os.path.join(MODELS_DIR, "density_thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\nSaved retrained model to {MODELS_DIR}/crowd_count_model.joblib")
    print("Restart the Flask app to use it.")


if __name__ == "__main__":
    main()
