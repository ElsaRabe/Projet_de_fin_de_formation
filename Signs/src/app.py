"""Interface Streamlit pour la reconnaissance de signes LSF en temps réel."""

import json
from collections import deque

import cv2
import keras
import numpy as np
import streamlit as st

from config import (
    MODEL_PATH,
    LABELS_PATH,
    SEQUENCE_LENGTH,
    CONFIDENCE_THRESHOLD,
)
from utils import create_hands_detector, draw_landmarks, extract_frame_features, open_webcam


@st.cache_resource
def load_model_and_labels():
    """Charge le modèle et les labels une seule fois (mis en cache par Streamlit)."""
    if not MODEL_PATH.exists():
        st.error(f"Modèle introuvable : {MODEL_PATH}. Lance train.py avant de continuer.")
        st.stop()

    model = keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        raw_labels = json.load(f)
    labels = {int(k): v for k, v in raw_labels.items()}
    return model, labels


def main() -> None:
    st.set_page_config(page_title="Traducteur LSF", layout="wide")
    st.title("🤟 Traducteur de langue des signes (LSF)")
    st.caption("Reconnaissance en temps réel de signes statiques via la webcam — MVP")

    model, labels = load_model_and_labels()

    if "running" not in st.session_state:
        st.session_state.running = False

    col_video, col_info = st.columns([2, 1])

    with col_info:
        start_button = st.button("▶️ Démarrer la webcam", use_container_width=True)
        stop_button = st.button("⏹️ Arrêter", use_container_width=True)
        st.divider()
        prediction_placeholder = st.empty()
        st.divider()
        st.subheader("Signes reconnus")
        st.write(", ".join(labels.values()))

    with col_video:
        video_placeholder = st.empty()

    if start_button:
        st.session_state.running = True
    if stop_button:
        st.session_state.running = False

    if not st.session_state.running:
        prediction_placeholder.info("En attente... clique sur *Démarrer la webcam*.")
        return

    try:
        cap = open_webcam()
    except RuntimeError as e:
        st.error(str(e))
        st.session_state.running = False
        return

    hands_detector = create_hands_detector()
    buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

    try:
        while st.session_state.running:
            success, frame = cap.read()
            if not success:
                st.error("Erreur de lecture de la webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(rgb_frame)

            hand_detected = bool(results.multi_hand_landmarks)
            if hand_detected:
                for hand_landmarks in results.multi_hand_landmarks:
                    draw_landmarks(frame, hand_landmarks)
                buffer.append(extract_frame_features(results))
            else:
                buffer.clear()

            buffer_ready = len(buffer) == SEQUENCE_LENGTH
            predicted_sign = None
            confidence = 0.0

            if buffer_ready:
                sequence = np.expand_dims(np.array(buffer, dtype=np.float32), axis=0)
                probabilities = model.predict(sequence, verbose=0)[0]
                class_index = int(np.argmax(probabilities))
                confidence = float(probabilities[class_index])
                if confidence >= CONFIDENCE_THRESHOLD:
                    predicted_sign = labels[class_index]

            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(display_frame, channels="RGB", use_container_width=True)

            if not hand_detected:
                prediction_placeholder.warning("Aucune main détectée")
            elif not buffer_ready:
                prediction_placeholder.info("Initialisation du buffer...")
            elif predicted_sign is not None:
                prediction_placeholder.success(f"**{predicted_sign}**  —  {confidence * 100:.0f}%")
            else:
                prediction_placeholder.warning(f"Signe incertain  —  {confidence * 100:.0f}%")

    finally:
        cap.release()
        hands_detector.close()


if __name__ == "__main__":
    main()