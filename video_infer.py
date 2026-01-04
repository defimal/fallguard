import cv2
import tensorflow as tf
import numpy as np

IMG_SIZE = 224
CLASS_NAMES = ["fall", "stand"]

# Load trained model
model = tf.keras.models.load_model("fallguard_model", compile=False)
print("Model loaded")

def preprocess_frame(frame):
    h, w, _ = frame.shape
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img, w, h

def draw_prediction(frame, bbox, label, confidence):
    h, w, _ = frame.shape
    xmin, ymin, xmax, ymax = bbox

    xmin = int(xmin * w)
    ymin = int(ymin * h)
    xmax = int(xmax * w)
    ymax = int(ymax * h)

    color = (0, 0, 255) if label == "fall" else (0, 255, 0)

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    text = f"{label} ({confidence:.2f})"
    cv2.putText(frame, text, (xmin, max(20, ymin - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame

# Open video
cap = cv2.VideoCapture("test.mp4")  # make sure this exists

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img, w, h = preprocess_frame(frame)

    preds = model.predict(img, verbose=0)
    bbox = preds["bbox"][0]
    class_probs = preds["class"][0]

    class_id = np.argmax(class_probs)
    confidence = class_probs[class_id]
    label = CLASS_NAMES[class_id]

    # Confidence threshold
    if confidence > 0.7:
        frame = draw_prediction(frame, bbox, label, confidence)

    cv2.imshow("FallGuard", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
