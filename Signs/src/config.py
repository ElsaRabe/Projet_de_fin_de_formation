"""Configuration centralisée du projet.

Toute constante partagée entre plusieurs modules doit vivre ici.
Ajouter un nouveau signe ne nécessite de modifier QUE ce fichier.
"""

from pathlib import Path

# --- Chemins ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"


CURRENT_MODEL_VERSION = "v1"
CURRENT_MODEL_DIR = MODELS_DIR / CURRENT_MODEL_VERSION
LABELS_PATH = CURRENT_MODEL_DIR / "labels.json"
MODEL_PATH = CURRENT_MODEL_DIR / "model.keras"

# --- Signes reconnus (MVP) ---
# Pour ajouter un signe : l'ajouter ici, relancer collect_data.py puis train.py.
SIGNS: list[str] = [
    "salut",
    "merci",
    "oui",
    "non",
    "sil_vous_plait",
    "au_revoir",
]

# --- MediaPipe Hands ---
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# --- Landmarks / features ---
LANDMARKS_PER_HAND = 21          # nombre de points fournis par MediaPipe Hands
COORDS_PER_LANDMARK = 3          # x, y, z
FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK   # 63
FEATURE_VECTOR_SIZE = FEATURES_PER_HAND * MAX_NUM_HANDS         # 126 (2 mains)

# --- Collecte de données (séquences dynamiques) ---
SEQUENCE_LENGTH = 30              # nombre de frames par séquence
SAMPLES_PER_SIGN = 50             # nombre de séquences à collecter par signe
COUNTDOWN_SECONDS = 2             # pause entre deux enregistrements

# --- Entraînement ---
TRAIN_TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42
EPOCHS = 100
BATCH_SIZE = 8
EARLY_STOPPING_PATIENCE = 15

# --- Prédiction ---
CONFIDENCE_THRESHOLD = 0.7        # score minimum pour afficher une prédiction