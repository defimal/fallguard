Fall Detection Web App

A computer vision web application that detects human falls in video using a Roboflow model and visualizes predictions with bounding boxes. The app is built with Streamlit and OpenCV and allows users to upload videos, run inference, and download annotated results.

Features
- Upload video files for fall detection
- Frame-by-frame inference using Roboflow
- Bounding box visualization with confidence scores
- Adjustable confidence threshold
- Optimized inference using frame stride
- Downloadable processed (annotated) video
- Secure API key handling using environment variables
- Deployed as a web application

Model Details Builde by Akasha Ahmad
Task: Human fall detection
Classes: fall, stand
Inference: Roboflow hosted inference API
Bounding boxes: center-based (x, y, width, height) format

Note: Labels are swapped in the UI to match the demonstration logic used during development.

Tech Stack
Frontend / UI: Streamlit
Computer Vision: OpenCV (headless)
Inference: Roboflow Inference SDK
Language: Python
Deployment: Streamlit Community Cloud (or similar platform)

Project Structure
scan_video_classes.py   – Main Streamlit application
requirements.txt        – Python dependencies
runtime.txt             – Python version (3.11)
packages.txt            – System packages for OpenCV
README.md               – Documentation

Local Setup
1. Clone the repository
git clone https://github.com/your-username/fall-detection-app.git
cd fall-detection-app

2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Set the Roboflow API key
export ROBOFLOW_KEY="your_roboflow_api_key"

5. Run the app
streamlit run scan_video_classes.py

The app will be available at
https://fall-guard.streamlit.app/?

Deployment Notes
- The Roboflow API key must be stored as an environment variable or secret.
- Do not hardcode API keys in the source code.
- For Streamlit Community Cloud, add the following under App Settings → Secrets:
ROBOFLOW_KEY = "your_roboflow_api_key"

Implementation Notes
- Uses opencv-python-headless for server compatibility.
- Output video is written to disk before being offered for download.
- Frame dimensions and FPS are matched to the input video to prevent corrupted output.
- GUI functions such as cv2.imshow() are not used to ensure headless deployment compatibility.

Future Improvements
- Temporal smoothing to reduce flickering detections
- Fall-alert logic based on consecutive frames
- Export detection metadata to CSV
- Performance optimizations and batching
- Real-time webcam streaming with WebRTC

Author
Defi Maleji
Computer Science Student – Cybersecurity & Software Engineering

Credits

This project uses the "Fall Detection" model created by Akasha Ahmad and hosted on Roboflow.
Model URL: https://universe.roboflow.com/akasha-ahmad-w0jif/fall-detection-mbldh/model/1
The model is accessed via Roboflow’s inference API and is used for educational and demonstration purposes.

