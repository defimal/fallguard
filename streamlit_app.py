import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import tempfile

IMG_SIZE = 224
CLASS_NAMES = ["fall", "stand"]

st.set_page_config(page_title="FallGuard", layout="centered")
st.title("FallGuard — Fall Detection System")
st.info(
    "⚠️ Live demo runs on CPU (Streamlit Community Cloud). "
    "Video inference may be slower than local GPU execution. "
    "See demo video for full-speed performance."
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fallguard_model", compile=False)

model = load_model()

def preprocess_frame(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------
# INPUT MODE SELECTION
# -------------------------
mode = st.radio(
    "Choose input type",
    ["Video Upload", "Take a Picture"]
)

# =========================
# IMAGE MODE
# =========================
if mode == "Take a Picture":
    st.subheader("Capture or Upload an Image")

    img_file = st.camera_input("Take a photo")

    if img_file is None:
        img_file = st.file_uploader(
            "Or upload an image",
            type=["jpg", "jpeg", "png"]
        )

    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img = preprocess_frame(frame)
        preds = model.predict(img, verbose=0)

        bbox = preds["bbox"][0]
        class_probs = preds["class"][0]

        class_id = np.argmax(class_probs)
        confidence = class_probs[class_id]
        label = CLASS_NAMES[class_id]

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
            0.8,
            color,
            2
        )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption="Prediction Result", width="stretch")



# =========================
# VIDEO MODE
# =========================
if mode == "Video Upload":
    uploaded_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        threshold = st.slider(
            "Confidence Threshold",
            0.3,
            0.9,
            0.7
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

            if confidence > threshold:
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

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, width="stretch")


        cap.release()
