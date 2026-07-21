"""
Train a crowd-counting model:
  image -> engineered features (features.py) -> RandomForestRegressor -> predicted count
  predicted count -> tertile thresholds -> Low / Medium / High density label

Run:  python3 train_model.py
Outputs (in ./out):
  crowd_count_model.joblib   - trained sklearn Pipeline (StandardScaler + RandomForestRegressor)
  density_thresholds.json    - {"low_max": .., "medium_max": ..} cut points for the 3 classes
  training_report.json       - MAE / R2 / feature importances / dataset info
  sample_scenes/             - a handful of example generated scenes for sanity-checking
"""
import json
import os
import time

import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from synth_data import generate_scene
from features import extract_features, FEATURE_NAMES

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "sample_scenes"), exist_ok=True)

N_SAMPLES = int(os.environ.get("N_SAMPLES", 900))
MAX_COUNT = int(os.environ.get("MAX_COUNT", 150))
SEED = 42


def build_dataset(n_samples, max_count, seed=SEED, save_samples=6, log_every=100):
    rng = np.random.default_rng(seed)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    X = np.zeros((n_samples, len(FEATURE_NAMES)), dtype=np.float64)
    y = np.zeros(n_samples, dtype=np.float64)

    t0 = time.time()
    for i in range(n_samples):
        img, count = generate_scene(count=None, max_count=max_count, rng=rng)
        X[i] = extract_features(img, hog_detector=hog)
        y[i] = count

        if i < save_samples:
            cv2.imwrite(os.path.join(OUT_DIR, "sample_scenes", f"sample_{i}_count{count}.jpg"), img)

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"  generated {i + 1}/{n_samples}  ({elapsed:.1f}s elapsed)")

    return X, y


def main():
    print(f"Building synthetic dataset: {N_SAMPLES} images, count range 0-{MAX_COUNT} ...")
    X, y = build_dataset(N_SAMPLES, MAX_COUNT)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    print("Training RandomForestRegressor ...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)

    pred = np.clip(model.predict(X_test), 0, None)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"Test MAE: {mae:.2f} people   |   R2: {r2:.3f}")

    # Density thresholds: tertiles of the *true* count distribution in the
    # full dataset -> Low / Medium / High each cover ~1/3 of typical scenes.
    q1, q2 = np.quantile(y, [1 / 3, 2 / 3])
    thresholds = {"low_max": float(round(q1)), "medium_max": float(round(q2))}
    print(f"Density thresholds -> Low: 0-{thresholds['low_max']:.0f}, "
          f"Medium: {thresholds['low_max']:.0f}-{thresholds['medium_max']:.0f}, "
          f"High: {thresholds['medium_max']:.0f}+")

    importances = dict(zip(FEATURE_NAMES, model.named_steps["rf"].feature_importances_.round(4).tolist()))

    joblib.dump(model, os.path.join(OUT_DIR, "crowd_count_model.joblib"))
    with open(os.path.join(OUT_DIR, "density_thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(os.path.join(OUT_DIR, "training_report.json"), "w") as f:
        json.dump({
            "n_samples": N_SAMPLES,
            "max_count": MAX_COUNT,
            "test_mae": mae,
            "test_r2": r2,
            "density_thresholds": thresholds,
            "feature_importances": importances,
            "feature_names": FEATURE_NAMES,
            "note": ("Trained on procedurally generated synthetic crowd scenes "
                     "(see synth_data.py) because this environment has no internet "
                     "access to download a real crowd-counting dataset. Swap in real "
                     "data via build_dataset_from_real_images() in train_on_real_data.py "
                     "for production accuracy."),
        }, f, indent=2)

    print(f"\nSaved model + thresholds + report to {OUT_DIR}/")


if __name__ == "__main__":
    main()
