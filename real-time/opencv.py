import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# Load trained CNN model
model = load_model("../models/sign_language_model.h5")

# Load label mapping and reverse it
with open("../models/label_map.json", "r") as f:
    label_dict = json.load(f)
index_to_label = {v: k for k, v in label_dict.items()}

# OpenCV setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw a rectangle to guide hand placement
    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crop and preprocess ROI
    roi = frame[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (64, 64))           # Resize to training size
    roi_normalized = roi_resized / 255.0              # Normalize (0-1)
    roi_input = np.expand_dims(roi_normalized, axis=0)

    # Predict
    pred = model.predict(roi_input)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)
    class_label = index_to_label[class_idx]

    # Show prediction
    text = f"{class_label.upper()} ({confidence:.2f})"
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
