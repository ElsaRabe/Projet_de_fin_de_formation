"""Prétraitement des séquences collectées + entraînement du modèle LSTM.

Charge data/raw/<signe>/*.npy, entraîne un LSTM empilé, sauvegarde le
meilleur modèle et le mapping des labels dans models/<version>/.
"""

from email import header
import json

import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from config import (
    SIGNS,
    DATA_RAW_DIR,
    CURRENT_MODEL_DIR,
    MODEL_PATH,
    LABELS_PATH,
    SEQUENCE_LENGTH,
    FEATURE_VECTOR_SIZE,
    TRAIN_TEST_SPLIT_RATIO,
    RANDOM_SEED,
    EPOCHS,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
)


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Charge toutes les séquences depuis data/raw/ et construit X, y."""
    sequences: list[np.ndarray] = []
    labels: list[int] = []

    for label_index, sign in enumerate(SIGNS):
        sign_dir = DATA_RAW_DIR / sign
        if not sign_dir.exists():
            raise FileNotFoundError(
                f"Aucune donnée trouvée pour le signe '{sign}' ({sign_dir}). "
                "Lance collect_data.py avant l'entraînement."
            )

        sequence_files = sorted(sign_dir.glob("*.npy"))
        if not sequence_files:
            raise FileNotFoundError(f"Le dossier {sign_dir} ne contient aucune séquence .npy.")

        for seq_path in sequence_files:
            sequence = np.load(seq_path)
            if sequence.shape != (SEQUENCE_LENGTH, FEATURE_VECTOR_SIZE):
                raise ValueError(
                    f"{seq_path} a une forme invalide {sequence.shape}, "
                    f"attendu ({SEQUENCE_LENGTH}, {FEATURE_VECTOR_SIZE})."
                )
            sequences.append(sequence)
            labels.append(label_index)

        print(f"[OK] {sign} : {len(sequence_files)} séquences chargées.")

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y


def build_model(num_classes: int) -> keras.Model:
    """Construit un LSTM empilé adapté à la classification de séquences de gestes."""
    model = keras.Sequential([
        keras.layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_VECTOR_SIZE)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_labels() -> None:
    """Sauvegarde le mapping index -> nom du signe, dans l'ordre de config.SIGNS."""
    labels_map = {str(i): sign for i, sign in enumerate(SIGNS)}
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels_map, f, ensure_ascii=False, indent=2)
    print(f"[OK] Labels sauvegardés -> {LABELS_PATH}")


def evaluate(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Affiche la matrice de confusion et le rapport de classification sur le jeu de test."""
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print("\n--- Matrice de confusion (lignes = vrai, colonnes = prédit) ---")
    cm = confusion_matrix(y_test, y_pred, labels=range(len(SIGNS)))
    header = "        " + " ".join(f"{s[:6]:>6}" for s in SIGNS)
    print(header)
    for i, row in enumerate(cm):
        print(f"{SIGNS[i][:8]:>8}" + " ".join(f"{v:>6}" for v in row))

    print("\n--- Rapport de classification ---")
    print(classification_report(y_test, y_pred, target_names=SIGNS, zero_division=0))


def main() -> None:
    CURRENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("--- Chargement du dataset ---")
    X, y = load_dataset()
    print(f"\nDataset total : {X.shape[0]} séquences, {len(SIGNS)} classes.\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TRAIN_TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"Train : {X_train.shape[0]} séquences | Test : {X_test.shape[0]} séquences\n")

    model = build_model(num_classes=len(SIGNS))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    print("\n--- Entraînement ---")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    save_labels()
    print(f"[OK] Meilleur modèle sauvegardé -> {MODEL_PATH}")

    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()