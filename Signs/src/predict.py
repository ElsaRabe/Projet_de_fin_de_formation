"""Reconnaissance de signes en temps réel via la webcam.

Maintient un buffer glissant des dernières SEQUENCE_LENGTH frames et prédit
en continu. N'affiche un signe que si la confiance dépasse CONFIDENCE_THRESHOLD.
"""

import json
from collections import deque

import cv2
import keras
import numpy as np

from config import (
    MODEL_PATH,
    LABELS_PATH,
    SEQUENCE_LENGTH,
    CONFIDENCE_THRESHOLD,
)
from utils import create_hands_detector, draw_landmarks, extract_frame_features, open_webcam

WINDOW_NAME = "Reconnaissance LSF - temps reel"


def load_labels() -> dict[int, str]:
    """Charge le mapping index -> nom du signe depuis labels.json."""
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _draw_prediction(frame, sign: str | None, confidence: float, buffer_ready: bool, hand_detected: bool) -> None:
    """Affiche le signe reconnu et son score de confiance à l'écran."""
    if not hand_detected:
        text = "Aucune main detectee"
        color = (150, 150, 150)
    elif not buffer_ready:
        text = "Initialisation du buffer..."
        color = (200, 200, 200)
    elif sign is not None:
        text = f"{sign}  ({confidence * 100:.0f}%)"
        color = (0, 255, 0)
    else:
        text = f"... ({confidence * 100:.0f}%)"
        color = (0, 0, 255)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (30, 30, 30), -1)
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}. Lance train.py avant predict.py."
        )

    print("Chargement du modèle...")
    model = keras.models.load_model(MODEL_PATH)
    labels = load_labels()

    cap = open_webcam()
    hands_detector = create_hands_detector()
    buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

    print("Reconnaissance en cours. Appuie sur Échap pour quitter.\n")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Erreur de lecture de la webcam.")
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
                buffer.clear()  # réinitialise : on ne veut pas mélanger avant/après une absence de main

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

            _draw_prediction(frame, predicted_sign, confidence, buffer_ready, hand_detected)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                pass  # fenêtre pas encore prête, on ignore et on continue

    finally:
        cap.release()
        hands_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()