from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import base64
import pickle

# ================= CONFIG ================= #
MODEL_PATH = "faces_cnn_model.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"
CSV_PATH = "students.csv"  # Must have "Roll Number" and "Name" columns
DATASET_DIR = "Faces"      # Folder containing reference images
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ================= INIT ================= #
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= LOAD MODEL, ENCODER, DATA ================= #
model = load_model(MODEL_PATH)

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

students_df = pd.read_csv(CSV_PATH)

# ================= HELPER FUNCTIONS ================= #
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Read image in grayscale, resize, normalize, add batch dimension."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (100, 100))
    img_norm = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_norm, axis=-1)  # (100, 100, 1)
    img_input = np.expand_dims(img_input, axis=0)  # (1, 100, 100, 1)
    return img_input

def find_reference_image(predicted_label):
    """Find an image in DATASET_DIR where the label matches predicted_label."""
    for root, _, files in os.walk(DATASET_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                try:
                    label_from_file = f.split('_')[1].split('.')[0]
                    if str(label_from_file) == str(predicted_label):
                        return os.path.join(root, f)
                except IndexError:
                    continue
    return None

# ================= ROUTES ================= #
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    img_path = None

    # Handle file upload
    if file and allowed_file(file.filename):
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(img_path)

    # Handle camera capture
    elif request.form.get("image_data"):
        img_data = request.form.get("image_data").split(",")[1]
        img_bytes = base64.b64decode(img_data)
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], "captured_image.png")
        with open(img_path, "wb") as f:
            f.write(img_bytes)
    else:
        return render_template("results.html", error="No image provided")

    # Preprocess and predict
    img_input = preprocess_image(img_path)
    predictions = model.predict(img_input)
    predicted_class_index = int(np.argmax(predictions))
    confidence = round(float(predictions[0][predicted_class_index]) * 100, 2)
    print(f"Prediction Confidence: {confidence:.2f}%")

    # Map class index -> Roll Number using label encoder
    predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]

    # Get student info from CSV
    student_row = students_df[students_df["Roll Number"] == int(predicted_label)]
    if not student_row.empty:
        roll_number = student_row.iloc[0]["Roll Number"]
        name = student_row.iloc[0]["Name"]
    else:
        roll_number, name = None, None

    # Find reference image
    reference_image_path = find_reference_image(predicted_label) if roll_number else None
    ref_image_rel = None
    if reference_image_path and reference_image_path.startswith("static"):
        ref_image_rel = os.path.relpath(reference_image_path, "static")
    
    return render_template(
        "results.html",
        filename=os.path.basename(img_path),
        ref_image=ref_image_rel,
        roll_number=roll_number,
        name=name,
        confidence=confidence
    )

# ================= RUN ================= #
if __name__ == "__main__":
    app.run(debug=True)
