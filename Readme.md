# Smart Crowd Density Monitor

Flask app that estimates how many people are in a photo/webcam frame and
reports a **Low / Medium / High** density level.

## What's new: a trained crowd-counting model

The people **count** and **density level** are now produced by an actual
trained model, not a hardcoded rule:

```
image -> engineered features (ml/features.py) -> RandomForestRegressor -> predicted count
predicted count -> data-derived tertile thresholds -> Low / Medium / High
```

- **`ml/synth_data.py`** — procedurally generates crowd photos with a known
  ground-truth count (perspective scaling, overlap, lighting noise), used as
  training data.
- **`ml/features.py`** — extracts 10 holistic image features per photo (edge
  density at two scales, foreground/texture ratio, contour & corner counts,
  LBP texture entropy, plus the existing HOG detector's raw box count as one
  extra signal). This is the classical "holistic regression" approach to
  crowd counting (Chan & Vasconcelos, 2008 and successors) — it copes with
  occlusion/overlap in dense crowds far better than counting detector boxes.
- **`ml/train_model.py`** — trains a `StandardScaler + RandomForestRegressor`
  on 1,000 generated scenes. The shipped model (`models/crowd_count_model.joblib`)
  scored **MAE ≈ 4 people, R² ≈ 0.98** on held-out synthetic test data.
- **`models/density_thresholds.json`** — Low/Medium/High cut points, computed
  as tertiles of the training count distribution (currently Low 0–20, Medium
  20–52, High 52+), not hand-picked.
- **`crowd_model.py`** — loads the trained artifacts and exposes
  `predict_count()` / `classify_density()`, used by `app.py`.

The original YOLOv4-tiny/HOG detector is still used, but now only for the
detection-box overlay, the density heatmap, and the social-distancing check
— the headline count/density comes from the trained regressor instead.

### Important: synthetic training data

This model was trained on **procedurally generated** crowd scenes, not real
photos, because the environment that built this app has no internet access
to download a real crowd-counting dataset. The pipeline is completely real
and will work the same on real photos — it just needs real (image, count)
pairs to learn from for production-grade accuracy.

**To retrain on real photos:**

1. Collect crowd photos + a CSV of `filename,count` (see docstring at the
   top of `ml/train_on_real_data.py` for dataset suggestions — Mall dataset,
   ShanghaiTech, JHU-Crowd++, or just hand-counted photos of your own scenes).
2. Run:
   ```
   cd ml
   python3 train_on_real_data.py --images path/to/images --labels path/to/labels.csv
   ```
3. This overwrites `models/crowd_count_model.joblib` and
   `models/density_thresholds.json` in place. Restart the Flask app.

No GPU or deep learning framework needed — same lightweight scikit-learn
pipeline, just fed real images.

### Retraining the synthetic model with different settings

```
cd ml
N_SAMPLES=2000 MAX_COUNT=200 python3 train_model.py
```

Outputs go to `ml/out/` — copy `crowd_count_model.joblib` and
`density_thresholds.json` into `models/` to deploy them.

## Running the app

```
pip install -r requirements.txt
python3 app.py
```

Then open http://localhost:5000

## Features

- Upload an image or capture a webcam snapshot
- **Trained ML model** for people count + Low/Medium/High density
- Density heatmap overlay
- Social distancing violation detector
- Trend logging (CSV) + linear forecast chart
- One-click PDF report download
- Optional voice alert (text-to-speech, requires internet)
