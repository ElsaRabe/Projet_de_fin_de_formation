"""Fonctions réutilisables : détection des mains, extraction et normalisation
des landmarks (support 1 ou 2 mains, vecteur de taille fixe)."""

from __future__ import annotations

import numpy as np
import mediapipe as mp
import cv2

from config import (
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    FEATURES_PER_HAND,
    FEATURE_VECTOR_SIZE,
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

WRIST_INDEX = 0


def create_hands_detector() -> mp_hands.Hands:
    """Instancie un détecteur MediaPipe Hands avec la configuration du projet."""
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )


def _normalize_hand(hand_landmarks) -> np.ndarray:
    """Recentre les landmarks d'une main sur le poignet et les met à l'échelle.

    Rend la représentation invariante à la position de la main dans l'image
    et à sa distance par rapport à la caméra.
    """
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )

    wrist = coords[WRIST_INDEX].copy()
    coords -= wrist  # recentrage sur le poignet

    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale > 1e-6:
        coords /= scale  # mise à l'échelle par la taille de la main

    return coords.flatten()


def extract_frame_features(results) -> np.ndarray:
    """Construit un vecteur de features de taille fixe pour une frame.

    Gère 0, 1 ou 2 mains : les mains absentes sont remplies de zéros.
    Les mains sont ordonnées par label ('Left' avant 'Right') pour que
    la position dans le vecteur soit cohérente d'une frame à l'autre.
    """
    left_hand = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    right_hand = np.zeros(FEATURES_PER_HAND, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            label = handedness.classification[0].label  # "Left" ou "Right"
            normalized = _normalize_hand(hand_landmarks)
            if label == "Left":
                left_hand = normalized
            else:
                right_hand = normalized

    vector = np.concatenate([left_hand, right_hand])

    if vector.shape[0] != FEATURE_VECTOR_SIZE:
        raise ValueError(
            f"Vecteur de features invalide : {vector.shape[0]} valeurs "
            f"au lieu de {FEATURE_VECTOR_SIZE} attendues."
        )
    return vector


def draw_landmarks(frame, hand_landmarks) -> None:
    """Dessine les landmarks et connexions d'une main sur l'image (in-place)."""
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )


def open_webcam(index: int = 0) -> cv2.VideoCapture:
    """Ouvre la webcam et lève une erreur explicite si elle est inaccessible."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Impossible d'ouvrir la webcam (index={index}). "
            "Vérifie qu'aucune autre application ne l'utilise."
        )
    return cap