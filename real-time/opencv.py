# opencv_debug.py
import os
import time
import json
from collections import deque, Counter

import cv2
import numpy as np

# try/except model load so we can give a helpful error message
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except Exception as e:
    raise RuntimeError("TensorFlow / Keras not available in this environment. "
                       "Run this on your machine with TF installed.") from e

# ------------- Config -------------
MODEL_PATH_CANDIDATES = [
    "./sign_language_model.h5",
    "/mnt/data/sign_language_model.h5",
    "../models/sign_language_model.h5",
    "./models/sign_language_model.h5"
]
LABEL_PATH_CANDIDATES = [
    "./class_labels.json",
    "/mnt/data/class_labels.json",
    "../models/class_labels.json",
    "./models/class_labels.json"
]

USE_MOBILENET_PREPROCESS = True   # if False uses /255.0
MIRROR = False                    # set True if you want webcam mirrored
USE_SKIN_MASK = False             # toggle skin mask on/off
SMOOTHING = True
SMOOTH_WINDOW = 5
CONF_THRESHOLD = 0.45             # only show prediction if >= this
SAVE_LOW_CONF_FRAMES = True
LOW_CONF_SAVE_DIR = "./debug_lowconf"
# ----------------------------------

# find model
model_path = None
for p in MODEL_PATH_CANDIDATES:
    if os.path.exists(p):
        model_path = p
        break
if model_path is None:
    raise FileNotFoundError(f"Model not found. Checked: {MODEL_PATH_CANDIDATES}")
print("Loading model from:", model_path)
model = load_model(model_path)

# find labels
label_path = None
for p in LABEL_PATH_CANDIDATES:
    if os.path.exists(p):
        label_path = p
        break
if label_path is None:
    print("Warning: class_labels.json not found in checked paths.", LABEL_PATH_CANDIDATES)
    index_to_label = {}
else:
    with open(label_path, "r") as f:
        label_dict = json.load(f)
    # Build index->label mapping robustly
    try:
        # keys are strings -> indices: {"0":"a", "1":"b"}
        if all(str(k).lstrip("-").isdigit() for k in label_dict.keys()):
            index_to_label = {int(k): v for k, v in label_dict.items()}
        else:
            # otherwise assume {"a":0,"b":1}
            index_to_label = {int(v): k for k, v in label_dict.items()}
    except Exception as e:
        print("Could not parse label json. Raw:", label_dict)
        raise

print("Loaded label map (sample):", dict(list(index_to_label.items())[:10]))
num_labels = len(index_to_label)
model_outputs = int(model.output_shape[-1])
print("Model outputs:", model_outputs, "Label entries:", num_labels)
if num_labels != 0 and num_labels != model_outputs:
    print("WARNING: number of labels does not match model output units.")
    print("This mismatch will cause incorrect label display.")

# target size from model input
in_shape = model.input_shape  # (None, H, W, C)
target_h, target_w = int(in_shape[1]), int(in_shape[2])
TARGET_SIZE = (target_w, target_h)  # (W,H)
print("Using target size (W,H):", TARGET_SIZE)

# mediapipe
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                            min_detection_confidence=0.6, min_tracking_confidence=0.6)
except Exception as e:
    raise RuntimeError("MediaPipe is required for this script to detect hands. Install mediapipe.") from e

# webcam
cap = cv2.VideoCapture(0)
frame_buf = deque(maxlen=SMOOTH_WINDOW)

if SAVE_LOW_CONF_FRAMES and not os.path.exists(LOW_CONF_SAVE_DIR):
    os.makedirs(LOW_CONF_SAVE_DIR, exist_ok=True)

print("Controls: q=quit, f=toggle mirror, s=toggle skin-mask, m=toggle smoothing, p=toggle preprocess mode (mobile/0-1)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    if MIRROR:
        frame = cv2.flip(frame, 1)

    display = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    top3_text = ""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            margin = max(20, int(0.18 * max(x_max - x_min, y_max - y_min)))
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            hand_roi = frame[y_min:y_max, x_min:x_max]
            if hand_roi.size == 0:
                continue

            # optional skin mask
            if USE_SKIN_MASK:
                ycrcb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2YCrCb)
                lower = np.array([0, 133, 77], dtype=np.uint8)
                upper = np.array([255, 173, 127], dtype=np.uint8)
                mask = cv2.inRange(ycrcb, lower, upper)
                mask = cv2.medianBlur(mask, 5)
                hand_only = cv2.bitwise_and(hand_roi, hand_roi, mask=mask)
            else:
                hand_only = hand_roi

            # place on square black canvas
            hr, wr = hand_only.shape[:2]
            side = max(hr, wr)
            canvas = np.zeros((side, side, 3), dtype=np.uint8)
            y_off = (side - hr) // 2
            x_off = (side - wr) // 2
            canvas[y_off:y_off + hr, x_off:x_off + wr] = hand_only

            square_resized = cv2.resize(canvas, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            roi_rgb = cv2.cvtColor(square_resized, cv2.COLOR_BGR2RGB)

            # choose preprocessing
            if USE_MOBILENET_PREPROCESS:
                inp = preprocess_input(roi_rgb.astype("float32"))
            else:
                inp = roi_rgb.astype("float32") / 255.0

            # predict
            pred = model.predict(np.expand_dims(inp, axis=0), verbose=0)[0]
            top_idxs = pred.argsort()[-3:][::-1]
            top_labels = [(index_to_label.get(int(idx), str(idx)), float(pred[idx])) for idx in top_idxs]

            # smoothing buffer: store top-1 index & conf
            top1_idx = int(top_idxs[0])
            top1_conf = float(pred[top1_idx])
            frame_buf.append((top1_idx, top1_conf))

            # compute smoothed label
            if SMOOTHING and len(frame_buf) > 0:
                counts = Counter([x[0] for x in frame_buf])
                mode_idx, mode_count = counts.most_common(1)[0]
                # average confidence among frames that had mode_idx
                confs = [c for (i, c) in frame_buf if i == mode_idx]
                avg_conf = float(np.mean(confs)) if confs else 0.0
                display_idx = mode_idx
                display_conf = avg_conf
            else:
                display_idx = top1_idx
                display_conf = top1_conf

            display_label = index_to_label.get(int(display_idx), "?").upper()

            # if low confidence, optionally save ROI to disk for debugging
            if SAVE_LOW_CONF_FRAMES and display_conf < CONF_THRESHOLD:
                ts = int(time.time() * 1000)
                fn = os.path.join(LOW_CONF_SAVE_DIR, f"lowconf_{display_label}_{display_conf:.2f}_{ts}.png")
                cv2.imwrite(fn, square_resized)
                print("Saved low-conf ROI to", fn)

            # overlay text: show top 3
            top3_text = ", ".join([f"{lab.upper()}({conf:.2f})" for lab, conf in top_labels])
            cv2.putText(display, f"Top: {display_label} ({display_conf:.2f})",
                        (x_min, max(20, y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            cv2.putText(display, top3_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Sign Debug", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        MIRROR = not MIRROR
        print("Mirror:", MIRROR)
    elif key == ord('s'):
        USE_SKIN_MASK = not USE_SKIN_MASK
        print("Skin mask:", USE_SKIN_MASK)
    elif key == ord('m'):
        SMOOTHING = not SMOOTHING
        print("Smoothing:", SMOOTHING)
    elif key == ord('p'):
        USE_MOBILENET_PREPROCESS = not USE_MOBILENET_PREPROCESS
        print("Use MobileNet preprocess:", USE_MOBILENET_PREPROCESS)

cap.release()
cv2.destroyAllWindows()
hands.close()
