import os
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import cv2

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"
MODEL_PATH = BASE_DIR / "best.pt"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["RESULT_FOLDER"] = str(RESULT_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

# Load model
model = YOLO(str(MODEL_PATH))
model.to("cpu")  # force CPU


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def make_unique_name(filename: str) -> str:
    stem = secure_filename(Path(filename).stem)
    suffix = Path(filename).suffix.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{stem}_{timestamp}{suffix}"


@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    label_text = None
    error = None
    detections = []

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            error = "Please upload an image."
            return render_template("index.html", result_image=result_image, label_text=label_text, error=error, detections=detections)

        if not allowed_file(file.filename):
            error = "Only PNG, JPG, JPEG, and WEBP files are allowed."
            return render_template("index.html", result_image=result_image, label_text=label_text, error=error, detections=detections)

        filename = make_unique_name(file.filename)
        upload_path = UPLOAD_FOLDER / filename
        file.save(upload_path)

        # 🔥 Resize BEFORE inference (critical for memory)
        img = cv2.imread(str(upload_path))
        img = cv2.resize(img, (640, 640))
        cv2.imwrite(str(upload_path), img)

        # Run inference (lighter settings)
        results = model.predict(source=str(upload_path), imgsz=416, conf=0.25, verbose=False)
        r = results[0]

        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                name = model.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0].item())
                detections.append({
                    "label": name,
                    "confidence": round(conf * 100, 1),
                })

            top = detections[0]["label"].title()
            if len(detections) > 1:
                label_text = f"Detected {len(detections)} objects. Top prediction: {top}"
            else:
                label_text = f"Detected {top}"
        else:
            label_text = "No objects detected."

        # Save result image
        rendered = r.plot()
        result_name = f"result_{filename}"
        result_path = RESULT_FOLDER / result_name
        cv2.imwrite(str(result_path), rendered)

        result_image = url_for("static", filename=f"results/{result_name}")

    return render_template("index.html", result_image=result_image, label_text=label_text, error=error, detections=detections)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
