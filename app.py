"""
Smart Crowd Density Monitor - Flask Web App
Converted from the crowd_density_advanced.ipynb notebook.

Features:
- Upload an image OR capture a webcam snapshot in the browser
- HOG people detection (OpenCV built-in, pretrained - no training required)
- Density heatmap overlay
- Social distancing violation detector
- Trend logging (CSV) + linear forecast chart
- One-click PDF report download
- Optional voice alert (text-to-speech) generated as an MP3 you can play in-browser
"""

import os
import io
import base64
from datetime import datetime
from itertools import combinations

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for server-side rendering
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_file, jsonify, session
)
from werkzeug.utils import secure_filename

import crowd_model

# ----------------------------------------------------------------------
# App config
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "static", "results")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOG_FILE = os.path.join(BASE_DIR, "crowd_log.csv")

for d in (UPLOAD_DIR, RESULTS_DIR, REPORTS_DIR):
    os.makedirs(d, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB uploads

# ----------------------------------------------------------------------
# Core detectors (both pretrained - no training required)
#
# HOG alone badly undercounts dense/overlapping crowds (it's tuned for
# spaced-out pedestrians), which is why "Medium"/"High" density readings
# almost never showed up in practice from detector-box counts alone.
# YOLOv4-tiny has much better recall on crowded scenes and is used as the
# primary detector whenever its weights are available; HOG (with denser
# scan settings) is kept as an automatic offline fallback. Either way, the
# headline people-count and Low/Medium/High density label reported to the
# user come from crowd_model.py, a RandomForest regressor trained on
# holistic image features (see ml/train_model.py) that is more robust to
# occlusion in dense crowds than raw detector-box counting.
# ----------------------------------------------------------------------
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

YOLO_CFG_PATH = os.path.join(MODELS_DIR, "yolov4-tiny.cfg")
YOLO_WEIGHTS_PATH = os.path.join(MODELS_DIR, "yolov4-tiny.weights")
YOLO_NAMES_PATH = os.path.join(MODELS_DIR, "coco.names")

YOLO_FILES = {
    YOLO_CFG_PATH: "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    YOLO_WEIGHTS_PATH: "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
    YOLO_NAMES_PATH: "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
}

# Tuned HOG (denser scan than the OpenCV default -> catches more people,
# still undercounts very dense crowds, but far better than before).
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect_people_hog(image_bgr, resize_width=960, hit_threshold=0.0, nms_thresh=0.4):
    """Run the built-in HOG people detector on a BGR image.
    Returns (resized_bgr_image, boxes, centroids)."""
    h, w = image_bgr.shape[:2]
    scale = resize_width / w
    resized = cv2.resize(image_bgr, (resize_width, int(h * scale)))

    boxes, weights = hog.detectMultiScale(
        resized,
        winStride=(4, 4),
        padding=(8, 8),
        scale=1.03,
        hitThreshold=hit_threshold
    )

    if len(boxes) > 0:
        scores = [float(wt) for wt in weights.flatten()]
        keep_idx = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=scores,
            score_threshold=hit_threshold,
            nms_threshold=nms_thresh
        )
        keep_idx = keep_idx.flatten() if len(keep_idx) > 0 else []
        boxes = boxes[keep_idx]

    centroids = [(x + bw // 2, y + bh // 2) for (x, y, bw, bh) in boxes]
    return resized, boxes, centroids


def _download_yolo_files():
    """Download YOLOv4-tiny weights/cfg/names on first run if missing.
    Requires internet on the machine running this server. Returns True
    if all files are present/downloaded, False otherwise (offline)."""
    try:
        import urllib.request
        for path, url in YOLO_FILES.items():
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                urllib.request.urlretrieve(url, path)
        return all(os.path.exists(p) and os.path.getsize(p) > 0 for p in YOLO_FILES)
    except Exception as e:
        print(f"[detector] Could not download YOLOv4-tiny weights ({e}). Falling back to HOG.")
        return False


_yolo_net = None
_yolo_output_layers = None
_yolo_person_class_id = None
DETECTOR_NAME = "HOG (fallback)"

if _download_yolo_files():
    try:
        with open(YOLO_NAMES_PATH) as f:
            _coco_classes = [line.strip() for line in f]
        _yolo_person_class_id = _coco_classes.index("person")
        _yolo_net = cv2.dnn.readNetFromDarknet(YOLO_CFG_PATH, YOLO_WEIGHTS_PATH)
        _yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        _yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        _ln = _yolo_net.getLayerNames()
        _yolo_output_layers = [_ln[i - 1] for i in _yolo_net.getUnconnectedOutLayers().flatten()]
        DETECTOR_NAME = "YOLOv4-tiny"
        print("[detector] YOLOv4-tiny loaded — using it as the primary people detector.")
    except Exception as e:
        print(f"[detector] Failed to load YOLOv4-tiny ({e}). Falling back to HOG.")
        _yolo_net = None


def detect_people_yolo(image_bgr, resize_width=960, conf_thresh=0.3, nms_thresh=0.4, input_size=416):
    """Drop-in replacement for detect_people_hog() using YOLOv4-tiny.
    Returns (resized_bgr_image, boxes, centroids) in the same format."""
    h, w = image_bgr.shape[:2]
    scale = resize_width / w
    resized = cv2.resize(image_bgr, (resize_width, int(h * scale)))
    rh, rw = resized.shape[:2]

    blob = cv2.dnn.blobFromImage(resized, 1 / 255.0, (input_size, input_size), swapRB=True, crop=False)
    _yolo_net.setInput(blob)
    outputs = _yolo_net.forward(_yolo_output_layers)

    raw_boxes, confidences = [], []
    for out in outputs:
        for det in out:
            scores = det[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if class_id == _yolo_person_class_id and conf > conf_thresh:
                cx, cy, bw, bh = det[0] * rw, det[1] * rh, det[2] * rw, det[3] * rh
                x, y = int(cx - bw / 2), int(cy - bh / 2)
                raw_boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(conf))

    boxes = np.array(raw_boxes, dtype=int)
    if len(raw_boxes) > 0:
        keep_idx = cv2.dnn.NMSBoxes(raw_boxes, confidences, conf_thresh, nms_thresh)
        keep_idx = keep_idx.flatten() if len(keep_idx) > 0 else []
        boxes = boxes[keep_idx] if len(keep_idx) > 0 else np.empty((0, 4), dtype=int)

    centroids = [(int(x + bw / 2), int(y + bh / 2)) for (x, y, bw, bh) in boxes]
    return resized, boxes, centroids


def detect_people(image_bgr, resize_width=960):
    """Detect people using YOLOv4-tiny if it loaded successfully, otherwise HOG."""
    if _yolo_net is not None:
        return detect_people_yolo(image_bgr, resize_width=resize_width)
    return detect_people_hog(image_bgr, resize_width=resize_width)


def classify_density(count):
    """Bucket a people count into Low / Medium / High using the trained
    model's data-derived thresholds (see crowd_model.py / ml/train_model.py)."""
    return crowd_model.classify_density(count)


# ----------------------------------------------------------------------
# Percentage-of-capacity reporting
#
# The trained model still estimates a raw head count internally (it's the
# most reliable signal to feed the ML pipeline), but everything shown to
# the user - and everything logged/plotted - is expressed as a PERCENTAGE
# of the venue capacity the user enters for that image, not a headcount.
# ----------------------------------------------------------------------
DEFAULT_CAPACITY = 100  # used only if the user leaves the field blank/invalid

# % of capacity thresholds for the Low / Medium / High label shown to the user
PCT_LOW_MAX = 40
PCT_MEDIUM_MAX = 75


def count_to_percentage(count, capacity):
    if not capacity or capacity <= 0:
        capacity = DEFAULT_CAPACITY
    return round((count / capacity) * 100, 1)


def classify_density_percentage(pct):
    """Bucket a %-of-capacity figure into Low / Medium / High."""
    if pct <= PCT_LOW_MAX:
        level = "Low"
        suggestion = "Crowd is sparse relative to capacity. No action needed."
    elif pct <= PCT_MEDIUM_MAX:
        level = "Medium"
        suggestion = "Moderate occupancy. Keep monitoring the area."
    else:
        level = "High"
        suggestion = "Venue is near/over capacity. Consider crowd-control measures or limiting further entry."
    return level, suggestion


# ----------------------------------------------------------------------
# Feature 1 - Density Heatmap
# ----------------------------------------------------------------------
def build_heatmap(image_bgr, centroids, grid=32, blur_ksize=51):
    h, w = image_bgr.shape[:2]
    density_map = np.zeros((h, w), dtype=np.float32)

    for (cx, cy) in centroids:
        if 0 <= cy < h and 0 <= cx < w:
            density_map[cy, cx] += 1.0

    if blur_ksize % 2 == 0:
        blur_ksize += 1
    density_map = cv2.GaussianBlur(density_map, (blur_ksize, blur_ksize), 0)

    if density_map.max() > 0:
        density_map = density_map / density_map.max()

    heat_color = cv2.applyColorMap((density_map * 255).astype("uint8"), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.6, heat_color, 0.4, 0)
    return overlay


# ----------------------------------------------------------------------
# Feature 2 - Social Distancing Violation Detector
# ----------------------------------------------------------------------
def check_social_distancing(image_bgr, boxes, centroids, distance_factor=2.0):
    annotated = image_bgr.copy()
    violations = []

    if len(boxes) > 0:
        avg_width = np.mean([bw for (_, _, bw, _) in boxes])
        threshold = avg_width * distance_factor
    else:
        threshold = 0

    for i, j in combinations(range(len(centroids)), 2):
        (x1, y1), (x2, y2) = centroids[i], centroids[j]
        dist = np.hypot(x2 - x1, y2 - y1)
        if dist < threshold:
            violations.append((i, j))
            cv2.line(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for idx, (x, y, bw, bh) in enumerate(boxes):
        is_violator = any(idx in pair for pair in violations)
        color = (0, 0, 255) if is_violator else (0, 255, 0)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

    return annotated, len(violations)


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def log_reading(percentage, capacity, level, violations):
    # If a log from the old (headcount-based) schema exists, archive it
    # rather than mixing schemas.
    if os.path.exists(LOG_FILE):
        try:
            existing_cols = pd.read_csv(LOG_FILE, nrows=0).columns.tolist()
        except Exception:
            existing_cols = []
        if "percentage" not in existing_cols:
            backup_name = LOG_FILE.replace(".csv", f"_legacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            os.rename(LOG_FILE, backup_name)

    row = pd.DataFrame([{
        "timestamp": datetime.now(),
        "percentage": percentage,
        "capacity": capacity,
        "level": level,
        "violations": violations
    }])
    if os.path.exists(LOG_FILE):
        row.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(LOG_FILE, mode="w", header=True, index=False)


# ----------------------------------------------------------------------
# Feature 3 - Trend + Forecast (returns a base64 PNG for embedding in HTML)
# ----------------------------------------------------------------------
def trend_chart_base64(forecast_steps=5, very_high_threshold=None):
    if very_high_threshold is None:
        very_high_threshold = PCT_MEDIUM_MAX  # % of capacity that counts as "High"
    if not os.path.exists(LOG_FILE):
        return None, "No readings logged yet — analyze at least two images first."

    df = pd.read_csv(LOG_FILE, parse_dates=["timestamp"])
    if "percentage" not in df.columns:
        return None, "Log file is from an older version of the app. Clear the log and analyze at least two images to start fresh."
    if len(df) < 2:
        return None, "Need at least 2 logged readings to show a trend."

    x = np.arange(len(df))
    y = df["percentage"].values

    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs
    future_x = np.arange(len(df), len(df) + forecast_steps)
    future_y = slope * future_x + intercept

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y, "o-", label="Observed capacity used (%)")
    ax.plot(future_x, future_y, "o--", color="orange", label="Forecast")
    ax.axhline(very_high_threshold, color="red", linestyle=":",
               label=f"High threshold ({very_high_threshold}%)")
    ax.set_xlabel("Reading #")
    ax.set_ylabel("Capacity used (%)")
    ax.set_title("Crowd Density Trend & Forecast")
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    if slope > 0:
        steps_to_threshold = (very_high_threshold - y[-1]) / slope if slope != 0 else float("inf")
        if steps_to_threshold > 0:
            msg = (f"Trend is rising (~{slope:.2f}% of capacity/reading). At this rate, the "
                   f"'High' threshold could be reached in ~{steps_to_threshold:.1f} more readings.")
        else:
            msg = "Occupancy is already at or above the 'High' threshold."
    elif slope < 0:
        msg = f"Trend is falling (~{slope:.2f}% of capacity/reading). Crowd appears to be dispersing."
    else:
        msg = "Trend is flat — occupancy is steady."

    return img_b64, msg


# ----------------------------------------------------------------------
# Feature 5 - One-click PDF Report
# ----------------------------------------------------------------------
def generate_pdf_report(resized, heat_img, social_img, percentage, capacity, level, violations, suggestion):
    filename = f"crowd_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(REPORTS_DIR, filename)

    with PdfPages(filepath) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(cv2.cvtColor(social_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Detections")
        axes[0].axis("off")
        axes[1].imshow(cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Density Heatmap")
        axes[1].axis("off")

        fig.suptitle("Crowd Density Report", fontsize=16)
        fig.text(0.5, 0.02,
                  f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   |   "
                  f"Capacity Used: {percentage}% (of {capacity})   |   Density: {level}   |   Violations: {violations}\n"
                  f"Suggestion: {suggestion}",
                  ha="center", fontsize=10, wrap=True)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
            if "percentage" in df.columns and len(df) >= 2:
                fig2, ax2 = plt.subplots(figsize=(9, 5))
                ax2.plot(df["percentage"].values, "o-")
                ax2.set_title("Historical Capacity-Used Trend")
                ax2.set_xlabel("Reading #")
                ax2.set_ylabel("Capacity used (%)")
                pdf.savefig(fig2)
                plt.close(fig2)

    return filename


# ----------------------------------------------------------------------
# Feature 4 - Voice alert (optional, requires gTTS + internet)
# ----------------------------------------------------------------------
def speak_suggestion(text):
    """Generate an MP3 alert. Returns the filename (relative to static/results)
    or None if gTTS isn't available / fails (e.g. no internet)."""
    try:
        from gtts import gTTS
    except ImportError:
        return None
    try:
        filename = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        filepath = os.path.join(RESULTS_DIR, filename)
        tts = gTTS(text=text, lang="en")
        tts.save(filepath)
        return filename
    except Exception:
        return None


# ----------------------------------------------------------------------
# Full analysis pipeline
# ----------------------------------------------------------------------
def analyze_image(image_bgr, capacity=DEFAULT_CAPACITY, speak=False, save_pdf=False):
    resized, boxes, centroids = detect_people(image_bgr)

    # The trained regression model (crowd_model.py) still estimates a raw
    # head count internally from holistic image texture/edge features (it
    # handles dense/overlapping crowds better than counting detector boxes
    # alone) - but that headcount is only an intermediate value now. What's
    # reported to the user is that count expressed as a % of the venue
    # capacity they entered. The detector's boxes/centroids are still used
    # for the heatmap and social-distancing overlay below.
    ml_count = crowd_model.predict_count(resized, hog_detector=hog)
    count = ml_count if ml_count is not None else len(boxes)

    percentage = count_to_percentage(count, capacity)
    level, suggestion = classify_density_percentage(percentage)

    social_img, violations = check_social_distancing(resized, boxes, centroids)
    heat_img = build_heatmap(resized, centroids)

    if violations > 0:
        suggestion += f" Also detected {violations} pair(s) violating safe distancing."

    log_reading(percentage, capacity, level, violations)

    # Save annotated images to disk so they can be served to the browser
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    social_name = f"detections_{stamp}.jpg"
    heat_name = f"heatmap_{stamp}.jpg"
    cv2.imwrite(os.path.join(RESULTS_DIR, social_name), social_img)
    cv2.imwrite(os.path.join(RESULTS_DIR, heat_name), heat_img)

    result = {
        "percentage": percentage,
        "capacity": capacity,
        "level": level,
        "violations": violations,
        "suggestion": suggestion,
        "detections_img": social_name,
        "heatmap_img": heat_name,
        "audio_file": None,
        "pdf_file": None,
        "detector": DETECTOR_NAME,
        "model_used": crowd_model.MODEL_LOADED,
    }

    if speak:
        result["audio_file"] = speak_suggestion(suggestion)

    if save_pdf:
        result["pdf_file"] = generate_pdf_report(
            resized, heat_img, social_img, percentage, capacity, level, violations, suggestion
        )

    return result


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.context_processor
def inject_detector_name():
    return {"detector_name": DETECTOR_NAME, "model_loaded": crowd_model.MODEL_LOADED}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health_check():
    return jsonify({
        "status": "ok",
        "detector": DETECTOR_NAME,
        "model_loaded": crowd_model.MODEL_LOADED,
        "service": "crowd-density-monitor"
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """Handles both file-upload and webcam-snapshot (base64) submissions."""
    speak = request.form.get("speak") == "on"
    save_pdf = request.form.get("save_pdf") == "on"

    try:
        capacity = int(request.form.get("capacity", DEFAULT_CAPACITY))
        if capacity <= 0:
            raise ValueError
    except (TypeError, ValueError):
        capacity = DEFAULT_CAPACITY
        flash(f"Invalid or missing venue capacity — defaulted to {DEFAULT_CAPACITY}.")

    img_bgr = None

    # Case 1: file upload
    if "image_file" in request.files and request.files["image_file"].filename:
        file = request.files["image_file"]
        if not allowed_file(file.filename):
            flash("Unsupported file type. Please upload a PNG/JPG/BMP/WEBP image.")
            return redirect(url_for("index"))
        filename = secure_filename(file.filename)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = os.path.join(UPLOAD_DIR, f"{stamp}_{filename}")
        file.save(save_path)
        img_bgr = cv2.imread(save_path)
        if img_bgr is None or img_bgr.size == 0:
            flash("The uploaded image could not be read. Please try a different file.")
            return redirect(url_for("index"))

    # Case 2: webcam snapshot sent as base64 data URL
    elif request.form.get("webcam_data"):
        data_url = request.form["webcam_data"]
        try:
            header, encoded = data_url.split(",", 1)
            binary = base64.b64decode(encoded)
            arr = np.frombuffer(binary, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            flash("Could not read the webcam snapshot. Please try again.")
            return redirect(url_for("index"))

    if img_bgr is None or img_bgr.size == 0:
        flash("No image received. Please upload a file or capture a webcam snapshot.")
        return redirect(url_for("index"))

    result = analyze_image(img_bgr, capacity=capacity, speak=speak, save_pdf=save_pdf)
    return render_template("result.html", result=result)


@app.route("/trend")
def trend():
    img_b64, message = trend_chart_base64()
    return render_template("trend.html", img_b64=img_b64, message=message)


@app.route("/report/<filename>")
def download_report(filename):
    filepath = os.path.join(REPORTS_DIR, secure_filename(filename))
    if not os.path.exists(filepath):
        flash("Report not found.")
        return redirect(url_for("index"))
    return send_file(filepath, as_attachment=True)


@app.route("/log")
def view_log():
    if not os.path.exists(LOG_FILE):
        rows = []
    else:
        df = pd.read_csv(LOG_FILE)
        rows = df.to_dict("records")
    return render_template("log.html", rows=rows)


@app.route("/log/clear", methods=["POST"])
def clear_log():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        flash("Log cleared.")
    return redirect(url_for("view_log"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
