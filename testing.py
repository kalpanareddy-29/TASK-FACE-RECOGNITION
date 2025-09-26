import face_recognition
import numpy as np
from PIL import Image

def load_and_prepare_image(path, min_size=150):
    """
    Loads an image, converts it to RGB, upscales if too small, and returns a numpy array.
    """
    # Load image with PIL
    img = Image.open(path)
    
    # Convert to RGB (handles grayscale, RGBA, paletted images)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Upscale if image is too small
    width, height = img.size
    if width < min_size or height < min_size:
        scale_w = max(min_size / width, 1)
        scale_h = max(min_size / height, 1)
        new_size = (int(width * scale_w), int(height * scale_h))
        img = img.resize(new_size, Image.LANCZOS)
    
    # Convert to numpy array with uint8
    return np.array(img, dtype=np.uint8)

# Paths to images
known_path = r"C:\Users\kalpa\Desktop\TASK FACE RECOGNITON\Faces\5_1.jpg"
unknown_path = r"C:\Users\kalpa\Desktop\TASK FACE RECOGNITON\Faces\10_1.jpg"

# Load and prepare images
known_image = load_and_prepare_image(known_path)
unknown_image = load_and_prepare_image(unknown_path)

print("Known image shape:", known_image.shape)
print("Unknown image shape:", unknown_image.shape)

# Detect and encode faces
known_faces = face_recognition.face_encodings(known_image)
unknown_faces = face_recognition.face_encodings(unknown_image)

if not known_faces:
    print("❌ No face detected in the known image!")
elif not unknown_faces:
    print("❌ No face detected in the unknown image!")
else:
    known_encoding = known_faces[0]
    unknown_encoding = unknown_faces[0]

    # Compare faces
    if face_recognition.compare_faces([known_encoding], unknown_encoding)[0]:
        print("✅ Same person!")
    else:
        print("❌ Different person!")
