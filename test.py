from tensorflow.keras.models import load_model
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import random
import pickle

os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# ==== 1. Load model ====
model = load_model("faces_cnn_model.h5")
print("Model loaded successfully!")
# ==== 2. Load the label encoder ====
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("Model and label encoder loaded successfully!")

# ==== 3. Load CSV with roll numbers and names ====
students_df = pd.read_csv("students.csv")

# ==== 4. Upload test image ====
test_image_path = r"C:\Users\kalpa\Desktop\FACE\04.jpg"

# ==== 5. Preprocess test image ====
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
test_image_resized = cv2.resize(test_image, (100, 100))
test_image_norm = test_image_resized.astype('float32') / 255.0
test_input = np.expand_dims(test_image_norm, axis=-1)  # channel dim
test_input = np.expand_dims(test_input, axis=0)        # batch dim

# ==== 6. Predict ====
predictions = model.predict(test_input)
predicted_class_index = np.argmax(predictions)
confidence = predictions[0][predicted_class_index] * 100

predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
print(f"Predicted Label: {predicted_label}")

# ==== 7. Get matching student info from CSV ====
student_row = students_df[students_df['Roll Number'] == int(predicted_label)]
if not student_row.empty:
    roll_number = student_row.iloc[0]['Roll Number']
    name = student_row.iloc[0]['Name']
else:
    roll_number = None
    name = None

# ==== 8. Find actual/reference image from dataset ====
dataset_dir = "Faces"  # Folder where training images are stored
reference_image_path = None

for root, _, files in os.walk(dataset_dir):
    for f in files:
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            # Match file label to predicted label
            file_label = f.split('_')[1].split('.')[0]
            if file_label == str(predicted_label):
                reference_image_path = os.path.join(root, f)
                break
    if reference_image_path:
        break

if reference_image_path:
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
else:
    reference_image = None

# ==== 9. Display both images ====
plt.figure(figsize=(8, 4))

# Left: Reference image from dataset
if reference_image is not None:
    plt.subplot(1, 2, 1)
    plt.imshow(reference_image, cmap='gray')
    plt.title(f"Actual: {name if name else predicted_label}")
    plt.axis('off')

# Right: Uploaded test image
plt.subplot(1, 2, 2)
plt.imshow(test_image, cmap='gray')
if confidence >= 85.0:
    plt.title(f"Predicted: {name if name else predicted_label} ({confidence:.2f}%)")
else:
    plt.title(f"Unknown ({confidence:.2f}%)")
plt.axis('off')

plt.show()

# ==== 10. Print result in text ====
if confidence >= 85.0 and name:
    print(f"✅ Match Found — Roll Number: {roll_number}, Name: {name}")
    print(f"Confidence: {confidence:.2f}%")
else:
    print(f"❌ Unknown Face — Closest Match: {predicted_label} ({confidence:.2f}%)")
