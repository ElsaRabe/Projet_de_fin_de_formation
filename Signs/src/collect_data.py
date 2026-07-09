"""Collecte de séquences dynamiques pour chaque signe défini dans config.py.

Pour chaque signe et chaque séquence : un compte à rebours de préparation,
puis capture automatique de SEQUENCE_LENGTH frames. Chaque séquence est
sauvegardée dans data/raw/<signe>/<index>.npy avec la forme
(SEQUENCE_LENGTH, FEATURE_VECTOR_SIZE).
"""

import time

import cv2
import numpy as np

from config import (
    SIGNS,
    DATA_RAW_DIR,
    SAMPLES_PER_SIGN,
    SEQUENCE_LENGTH,
    COUNTDOWN_SECONDS,
)
from utils import create_hands_detector, draw_landmarks, extract_frame_features, open_webcam

WINDOW_NAME = "Collecte de sequences - LSF"


def _read_frame(cap, hands_detector):
    success, frame = cap.read()
    if not success:
        raise RuntimeError("Erreur de lecture de la webcam.")

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw_landmarks(frame, hand_landmarks)

    return frame, results


def _check_exit(key: int) -> None:
    if key == 27 or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        raise KeyboardInterrupt


def _countdown(cap, hands_detector, sign: str, seq_index: int) -> None:
    """Affiche un compte à rebours avant le début de la capture."""
    start = time.time()
    while time.time() - start < COUNTDOWN_SECONDS:
        frame, _ = _read_frame(cap, hands_detector)
        remaining = COUNTDOWN_SECONDS - (time.time() - start)

        cv2.putText(frame, f"Signe : {sign} ({seq_index + 1}/{SAMPLES_PER_SIGN})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Prepare-toi... {remaining:.1f}s",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        _check_exit(key)


def _capture_sequence(cap, hands_detector, sign: str, seq_index: int) -> np.ndarray:
    """Capture SEQUENCE_LENGTH frames et retourne la séquence de features."""
    sequence: list[np.ndarray] = []

    while len(sequence) < SEQUENCE_LENGTH:
        frame, results = _read_frame(cap, hands_detector)
        sequence.append(extract_frame_features(results))

        cv2.putText(frame, f"Signe : {sign} ({seq_index + 1}/{SAMPLES_PER_SIGN})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Enregistrement... {len(sequence)}/{SEQUENCE_LENGTH}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        _check_exit(key)

    return np.array(sequence, dtype=np.float32)


def main() -> None:
    cap = open_webcam()
    hands_detector = create_hands_detector()

    print(f"Signes a collecter : {SIGNS}")
    print(f"Sequences par signe : {SAMPLES_PER_SIGN} | Frames par sequence : {SEQUENCE_LENGTH}\n")

    try:
        for sign in SIGNS:
            sign_dir = DATA_RAW_DIR / sign
            sign_dir.mkdir(parents=True, exist_ok=True)

            existing = list(sign_dir.glob("*.npy"))
            if len(existing) >= SAMPLES_PER_SIGN:
                print(f"[SKIP] '{sign}' déjà collecté ({len(existing)} séquences).")
                continue

            print(f"--- Signe : {sign} ---")
            for seq_index in range(len(existing), SAMPLES_PER_SIGN):
                _countdown(cap, hands_detector, sign, seq_index)
                sequence = _capture_sequence(cap, hands_detector, sign, seq_index)

                output_path = sign_dir / f"{seq_index}.npy"
                np.save(output_path, sequence)
                print(f"[OK] Séquence {seq_index + 1}/{SAMPLES_PER_SIGN} sauvegardée -> {output_path}")

            print()

    except KeyboardInterrupt:
        print("\nCollecte interrompue par l'utilisateur.")
    finally:
        cap.release()
        hands_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()