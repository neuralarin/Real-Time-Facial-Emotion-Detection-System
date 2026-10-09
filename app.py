import av
import cv2
import mediapipe as mp
import streamlit as st
import torch
import torch.nn as nn
import re
from collections import deque, Counter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

MODEL_PATH = "model/emotion_classifier.pth"
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class EmotionDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 7),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource
def load_model():
    model = EmotionDetectionModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    return model.eval()


class EmotionProcessor(VideoProcessorBase):
    def __init__(self, facing_mode="user"):
        self.facing_mode = facing_mode
        self.model = load_model()
        self.predictions = deque(maxlen=15)
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.6
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.facing_mode == "user":
            img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if results.detections:
            h, w = img.shape[:2]
            box = max(
                results.detections,
                key=lambda d: d.location_data.relative_bounding_box.width
                * d.location_data.relative_bounding_box.height
            ).location_data.relative_bounding_box

            x0, y0 = max(0, int(box.xmin*w)), max(0, int(box.ymin*h))
            x1, y1 = min(w, int((box.xmin+box.width)*w)), min(h, int((box.ymin+box.height)*h))

            if x1 > x0 and y1 > y0:
                face = cv2.resize(rgb[y0:y1, x0:x1], (128, 128))
                tensor = torch.from_numpy(face.copy()).float().permute(2, 0, 1).unsqueeze(0)
                tensor = (tensor / 255.0 - 0.5) / 0.5

                with torch.no_grad():
                    probs = torch.softmax(self.model(tensor), dim=1)[0]
                    self.predictions.append(probs.argmax().item())

                emotion = Counter(self.predictions).most_common(1)[0][0]
                label = f"{CLASSES[emotion].capitalize()} ({probs[emotion].item()*100:.1f}%)"

                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(img, label, (x0, max(25, y0-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.set_page_config(page_title="Emotion Recognition", page_icon="😊")
st.title("😊 Real-Time Emotion Recognition")
st.write("Click START and allow camera access.")

user_agent = st.context.headers.get("User-Agent", "").lower()
is_mobile = bool(re.search(r"android|iphone|ipad|ipod|mobile", user_agent))

facing_mode = "user"

if is_mobile:
    camera = st.radio(
        "Choose camera",
        ["Front Camera 🤳", "Back Camera 📷"],
        horizontal=True,
    )
    facing_mode = "user" if camera == "Front Camera 🤳" else "environment"

if not hasattr(mp, "solutions"):
    st.error("Install compatible MediaPipe: pip install mediapipe==0.10.21")
else:
    webrtc_streamer(
        key=f"emotion-{facing_mode}",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: EmotionProcessor(facing_mode),
        media_stream_constraints={
            "video": {"facingMode": {"ideal": facing_mode}},
            "audio": False,
        },
        async_processing=True,
    )