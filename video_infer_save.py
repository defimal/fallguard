import cv2
import tensorflow as tf
import numpy as np

IMG_SIZE = 224
CLASS_NAMES = ["fall", "stand"]

model = tf.keras.models.load_model("fallguard_model", compile=False)
print("Model loaded")

def preprocess_frame(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

cap = cv2.VideoCapture("test.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "fallguard_output.mp4",
    fourcc,
    int(cap.get(cv2.CAP_PROP_FPS)),
    (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = preprocess_frame(frame)
    preds = model.predict(img, verbose=0)

    bbox = preds["bbox"][0]
    class_probs = preds["class"][0]

    class_id = np.argmax(class_probs)
    confidence = class_probs[class_id]
    label = CLASS_NAMES[class_id]

    if confidence > 0.7:
        h, w, _ = frame.shape
        xmin, ymin, xmax, ymax = bbox
        xmin, ymin = int(xmin * w), int(ymin * h)
        xmax, ymax = int(xmax * w), int(ymax * h)

        color = (0, 0, 255) if label == "fall" else (0, 255, 0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(
            frame,
            f"{label} ({confidence:.2f})",
            (xmin, max(20, ymin - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    out.write(frame)
    cv2.imshow("FallGuard", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
