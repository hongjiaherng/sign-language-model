import json
import math
import queue
import tempfile
import time
from collections import deque

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from streamlit_webrtc import WebRtcMode, webrtc_streamer


# =========================================================
# 1. MODEL ARCHITECTURE
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TemporalAttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return (weights * x).sum(dim=1)


class TransformerEncoderKeypoints(nn.Module):
    def __init__(self, num_classes, T=32, V=75, C=3, d_model=128):
        super().__init__()
        self.joint_embed = nn.Linear(C, 16)
        self.temporal_proj = nn.Linear(16 * V, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=T)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, 2)
        self.pool = TemporalAttentionPooling(d_model)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        B, C, T, V = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.joint_embed(x)
        x = x.reshape(B, T, -1)
        x = self.temporal_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.pool(x)
        return self.fc(x)


# =========================================================
# 2. PREPROCESSING LOGIC
# =========================================================
class Preprocessor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic

    def extract_xy(self, results):
        if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
            return None

        def _get(lms, n):
            if lms is None:
                return np.zeros((n, 3), dtype=np.float32)
            return np.array(
                [[lm.x, lm.y, lm.z] for lm in lms.landmark], dtype=np.float32
            )

        pose = _get(results.pose_landmarks, 33)
        lh = _get(results.left_hand_landmarks, 21)
        rh = _get(results.right_hand_landmarks, 21)

        return np.concatenate([pose, lh, rh], axis=0)

    def sample_and_pad(self, frames, target_frames=32):
        frames = np.array(frames)
        num_frames = frames.shape[0]

        if num_frames >= target_frames:
            indices = np.linspace(0, num_frames - 1, target_frames).astype(np.int32)
            output = frames[indices]
        else:
            last_frame = frames[-1][np.newaxis, :, :]
            pad = np.repeat(last_frame, target_frames - num_frames, axis=0)
            output = np.concatenate([frames, pad], axis=0)

        output = output.transpose(2, 0, 1)
        tensor = torch.FloatTensor(output).unsqueeze(0)
        return tensor


# =========================================================
# 3. VIDEO PROCESSOR (WebRTC)
# =========================================================
class VideoProcessor:
    def __init__(self):
        self.model = None
        self.device = None
        self.labels = None
        self.preprocessor = None
        self.threshold = 0.4
        self.frame_skip = 2

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,
        )

        self.sequence = deque(maxlen=32)
        self.frame_counter = 0
        self.current_prediction = "Waiting..."
        self.current_conf = 0.0
        self.result_queue = queue.Queue()

    def draw_hud(self, image, label, conf):
        h, w, _ = image.shape
        cv2.rectangle(image, (0, 0), (w, 50), (0, 0, 0), -1)
        if conf > 0:
            bar_width = int(w * conf)
            cv2.rectangle(image, (0, 45), (bar_width, 50), (0, 255, 0), -1)

        color = (0, 255, 0) if conf > 0.4 else (200, 200, 200)
        text = f"{label} ({conf * 100:.0f}%)" if label != "Waiting..." else "Waiting..."
        cv2.putText(image, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        self.frame_counter += 1

        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)
        image.flags.writeable = True

        mp.solutions.drawing_utils.draw_landmarks(
            image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS
        )

        if self.preprocessor:
            kp = self.preprocessor.extract_xy(results)
            if kp is not None:
                self.sequence.append(kp)

        if (
            self.model
            and len(self.sequence) == 32
            and (self.frame_counter % self.frame_skip == 0)
        ):
            try:
                tensor_kps = self.preprocessor.sample_and_pad(list(self.sequence)).to(
                    self.device
                )
                with torch.no_grad():
                    outputs = self.model(tensor_kps)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred_idx = torch.max(probs, 1)

                    this_conf = conf.item()
                    this_label = self.labels[pred_idx.item()]
                    self.current_conf = this_conf

                    if this_conf > 0.3:
                        self.current_prediction = this_label
                    else:
                        self.current_prediction = "Waiting..."

                    if not self.result_queue.full():
                        self.result_queue.put_nowait(
                            {
                                "label": self.current_prediction,
                                "conf": self.current_conf,
                                "probs": probs.cpu().numpy()[0],
                            }
                        )
            except Exception:
                pass

        self.draw_hud(image, self.current_prediction, self.current_conf)
        return av.VideoFrame.from_ndarray(image, format="bgr24")


# =========================================================
# 4. LOAD RESOURCES
# =========================================================
@st.cache_resource
def load_resources():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "checkpoints/best_model_kps_transformer.pth"
    GLOSS_PATH = "data/wlasl_reduced/gloss_map.json"

    try:
        with open(GLOSS_PATH, "r") as f:
            raw_glosses = json.load(f)

        if isinstance(raw_glosses, list):
            labels = {i: gloss for i, gloss in enumerate(raw_glosses)}
        else:
            sorted_pairs = sorted(raw_glosses.items(), key=lambda x: x[1])
            labels = (
                {v: k for k, v in sorted_pairs}
                if isinstance(list(raw_glosses.values())[0], int)
                else {
                    int(k): v
                    for k, v in sorted(raw_glosses.items(), key=lambda x: int(x[0]))
                }
            )

    except Exception:
        return None, None, None

    try:
        model = TransformerEncoderKeypoints(num_classes=len(labels))
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
    except Exception:
        return None, None, None

    return model, labels, device


# =========================================================
# 5. MAIN APP UI
# =========================================================
st.set_page_config(page_title="WLASL Sign Recognition", layout="wide")

st.markdown(
    """
<style>
    /* Center align main content and ensure width */
    .main .block-container {
        max-width: 1400px;
        padding-left: 3rem;
        padding-right: 3rem;
        padding-top: 2rem;
    }
    
    /* Center title */
    h1 {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Center columns container */
    [data-testid="column"] {
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    /* Style classes */
    .big-label { 
        font-size: 20px !important; 
        color: #666; 
        margin-bottom: -5px; 
        text-align: center;
        width: 100%;
    }
    .big-pred { 
        font-size: 42px !important; 
        font-weight: bold; 
        margin-top: 0px; 
        text-align: center;
        width: 100%;
    }
    .big-conf { 
        font-size: 18px !important; 
        font-weight: bold; 
        text-align: center;
        width: 100%;
    }
    .color-green { color: #2ca02c; }
    .color-gray { color: #888888; }
    
    /* Center file uploader */
    .stFileUploader {
        text-align: center;
        width: 100%;
    }
    
    /* Center radio buttons */
    .stRadio > div {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
    }
    
    /* Center sliders */
    .stSlider {
        text-align: center;
        width: 100%;
    }
    
    /* Ensure video containers are wide */
    [data-testid="stWebRTC"] {
        width: 100%;
        max-width: 100%;
    }
    
    /* Center buttons */
    button {
        margin: 0 auto;
        display: block;
    }
    
    /* Center images */
    .stImage {
        text-align: center;
        margin: 0 auto;
    }
    
    /* Center progress bars */
    .stProgress {
        width: 100%;
    }
    
    /* Center charts */
    [data-testid="stBarChart"] {
        width: 100%;
    }
    
    /* Ensure headings are centered */
    h3, h4 {
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🤟 WLASL-35 Sign Language Recognizer")

model, labels, device = load_resources()
preprocessor = Preprocessor()
if not model:
    st.stop()

# --- TOP SETTINGS ---
col_set1, col_set2, col_set3 = st.columns([1, 1, 1])
with col_set1:
    mode = st.radio("Input Mode", ["Video Upload", "Live Webcam"])
with col_set2:
    ui_threshold = st.slider("Success Threshold", 0.0, 1.0, 0.5)
with col_set3:
    st.empty()  # Spacer for centering

st.markdown("---")

# --- LIVE WEBCAM ---
if mode == "Live Webcam":
    col_cam, col_stats = st.columns([1, 1], gap="large")
    with col_cam:

        def video_frame_callback_factory():
            processor = VideoProcessor()
            processor.model = model
            processor.device = device
            processor.labels = labels
            processor.preprocessor = preprocessor
            return processor

        ctx = webrtc_streamer(
            key="wlasl-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            video_processor_factory=video_frame_callback_factory,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_stats:
        ph_label_container = st.empty()
        ph_conf_container = st.empty()
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Top Predictions")
        ph_chart = st.empty()

    if ctx.state.playing:
        while True:
            try:
                if ctx.video_processor:
                    result = ctx.video_processor.result_queue.get(timeout=1.0)

                    label = result["label"]
                    conf = result["conf"]

                    if conf > ui_threshold and label != "Waiting...":
                        color_class = "color-green"
                    else:
                        color_class = "color-gray"

                    ph_label_container.markdown(
                        f"""
                        <div class="big-label">Detected Sign:</div>
                        <div class="big-pred {color_class}">{label}</div>
                    """,
                        unsafe_allow_html=True,
                    )

                    ph_conf_container.markdown(
                        f"""
                        <div class="big-conf {color_class}">Confidence: {conf * 100:.1f}%</div>
                    """,
                        unsafe_allow_html=True,
                    )

                    probs = result["probs"]
                    top5 = probs.argsort()[-5:][::-1]
                    chart_data = {labels[i]: probs[i] for i in top5}
                    ph_chart.bar_chart(chart_data)

            except queue.Empty:
                pass
            except Exception:
                break
            time.sleep(0.05)

# --- VIDEO UPLOAD ---
elif mode == "Video Upload":
    # 1. Layout Columns
    col_video, col_results = st.columns([1, 1], gap="large")

    with col_video:
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        # UI Elements
        with col_video:
            st_video_placeholder = st.empty()
            progress_bar = st.progress(0)

        with col_results:
            ph_label_ul = st.empty()
            ph_conf_ul = st.empty()
            st.markdown("#### Top Predictions")
            ph_chart_ul = st.empty()

            # Place the Replay button here, but enable it only after processing
            st_replay_placeholder = st.empty()

        # 2. Process Video
        valid_frames = []
        original_frames = []
        holistic = mp.solutions.holistic.Holistic(static_image_mode=False)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        curr = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save original for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frames.append(rgb_frame)

            # Extract
            frame.flags.writeable = False
            results = holistic.process(rgb_frame)
            kp = preprocessor.extract_xy(results)
            if kp is not None:
                valid_frames.append(kp)

            curr += 1
            if total > 0:
                progress_bar.progress(min(curr / total, 1.0))

        holistic.close()
        cap.release()

        # 3. Inference
        if len(valid_frames) > 0:
            tensor_in = preprocessor.sample_and_pad(valid_frames).to(device)
            with torch.no_grad():
                outputs = model(tensor_in)
                probs = torch.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, 1)

            p_gloss = labels[pred_idx.item()]
            p_conf = conf.item()
            p_probs = probs.cpu().numpy()[0]

            # 4. Update Result UI (Big Text & Chart)
            if p_conf > ui_threshold:
                color_class = "color-green"
            else:
                color_class = "color-gray"

            ph_label_ul.markdown(
                f"""
                <div class="big-label">Detected Sign:</div>
                <div class="big-pred {color_class}">{p_gloss}</div>
            """,
                unsafe_allow_html=True,
            )

            ph_conf_ul.markdown(
                f"""
                <div class="big-conf {color_class}">Confidence: {p_conf * 100:.1f}%</div>
            """,
                unsafe_allow_html=True,
            )

            top5 = p_probs.argsort()[-5:][::-1]
            chart_data = {labels[i]: p_probs[i] for i in top5}
            ph_chart_ul.bar_chart(chart_data)

            # 5. Prepare Annotated Frames
            annotated_frames = []
            for frame in original_frames:
                frame_copy = frame.copy()
                # Overlay
                frame_copy = cv2.resize(frame_copy, (640, 480))
                cv2.rectangle(frame_copy, (0, 0), (640, 50), (0, 0, 0), -1)
                cv2.putText(
                    frame_copy,
                    f"{p_gloss} ({p_conf * 100:.0f}%)",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                annotated_frames.append(frame_copy)

            # 6. Play & Replay Logic
            # We use Session State to handle replay triggers
            if "replay_trigger" not in st.session_state:
                st.session_state.replay_trigger = 0

            # Replay Button
            if st_replay_placeholder.button("Replay Prediction"):
                st.session_state.replay_trigger += 1

            # Play Loop
            for frame in annotated_frames:
                st_video_placeholder.image(frame)
                time.sleep(0.03)  # Approx 30 FPS

        else:
            st.error("No hands detected in video.")
