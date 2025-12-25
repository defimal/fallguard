# streamlit_app.py
# Run:
#   export ROBOFLOW_KEY="your_key_here"
#   streamlit run streamlit_app.py

import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from inference_sdk import InferenceHTTPClient

MODEL_ID = "fall-detection-mbldh/1"

@st.cache_resource
def get_client():
    api_key = os.getenv("ROBOFLOW_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_KEY not set (export it in your terminal)")
    return InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=api_key)

def annotate_frame(frame_bgr, result, conf_threshold: float):
    for pred in result.get("predictions", []):
        confidence = float(pred.get("confidence", 0.0))
        if confidence < conf_threshold:
            continue

        x = int(pred["x"])
        y = int(pred["y"])
        w = int(pred["width"])
        h = int(pred["height"])

        class_name = str(pred["class"]).lower()

        # Swap labels (display only)
        if class_name == "fall":
            class_name = "stand"
        elif class_name == "stand":
            class_name = "fall"

        # Color mapping
        color = (0, 0, 255) if class_name == "fall" else (0, 255, 0)

        # Bounding box (friend-style center coords)
        cv2.rectangle(
            frame_bgr,
            (x - w // 2, y - h // 2),
            (x + w // 2, y + h // 2),
            color,
            5,
        )

        # Label
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(
            frame_bgr,
            label,
            (x - w // 2, y - h // 2 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
        )
    return frame_bgr


st.title("Fall Detection Demo (Roboflow + Streamlit)")

conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
option = st.radio("Choose input source:", ["Upload Video", "Webcam"])

client = get_client()

if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "m4v"])
    stride = st.selectbox("Inference stride (every N frames)", [1, 2, 5, 10], index=2)

    if uploaded_file is not None:
        # Save upload
        suffix = "." + uploaded_file.name.split(".")[-1].lower()
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Could not open uploaded video.")
            st.stop()

        # Match output properties to input (friend-style)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps:
            fps = 30.0
        fps = float(fps)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            st.error("Invalid video dimensions.")
            st.stop()

        # Create output file (downloadable)
        out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out_path = out_tmp.name
        out_tmp.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            st.error("Failed to create output video. Try VLC for playback or install ffmpeg to re-encode.")
            st.stop()

        stframe = st.empty()
        progress = st.progress(0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_i = 0
        last_result = {"predictions": []}

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_i += 1

            # Safety: enforce exact size for the writer
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            if frame_i % stride == 0:
                last_result = client.infer(frame, model_id=MODEL_ID)

            annotated = frame.copy()
            annotate_frame(annotated, last_result, conf_threshold)

            # Live preview
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

            # Persist to output file (this enables download)
            writer.write(annotated)

            if total_frames:
                progress.progress(min(1.0, frame_i / total_frames))

        cap.release()
        writer.release()

        st.success("Processing complete.")

        # Optional preview of final video (may not play in some browsers depending on codec)
        st.video(out_path)

        # Download button (THIS is the fix)
        with open(out_path, "rb") as f:
            st.download_button(
                label="Download processed video",
                data=f,
                file_name="fall_detection_output.mp4",
                mime="video/mp4",
            )

else:
    st.write("Capture an image using your webcam.")
    camera_input = st.camera_input("Take a picture")

    if camera_input is not None:
        file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR

        result = client.infer(frame, model_id=MODEL_ID)
        annotated = frame.copy()
        annotate_frame(annotated, result, conf_threshold)

        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

        # Download annotated image
        ok, buffer = cv2.imencode(".jpg", annotated)
        if ok:
            st.download_button(
                label="Download image",
                data=buffer.tobytes(),
                file_name="fall_detection.jpg",
                mime="image/jpeg",
            )
