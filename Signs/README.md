# Traducteur LSF – Reconnaissance de signes en temps réel

> MVP de reconnaissance de la **Langue des Signes Française (LSF)** utilisant la vision par ordinateur et le Deep Learning pour traduire des gestes en français en temps réel.

> **Version 1 (MVP)** : cette première version reconnaît **6 signes dynamiques** et constitue une base évolutive permettant d'ajouter facilement de nouveaux signes et fonctionnalités.

---

## Démonstration

> *(GIF ou des captures d'écran de l'application.)*

---

## Fonctionnalités

- Reconnaissance des signes en temps réel via une webcam.
- Détection des mains avec **MediaPipe Hands**.
- Extraction et normalisation des landmarks des mains.
- Classification des séquences de gestes grâce à un modèle **LSTM** développé avec TensorFlow/Keras.
- Affichage du signe reconnu avec un score de confiance.
- Interface utilisateur réalisée avec **Streamlit**.

---

## Signes reconnus (V1)

- Salut
- Merci
- Oui
- Non
- S'il vous plaît
- Au revoir

Ces signes sont **dynamiques** : le modèle apprend la trajectoire complète du mouvement sur une séquence d'environ **30 frames**, et non une simple posture de la main.

---

## Fonctionnement

```text
Webcam
    │
    ▼
OpenCV
    │
    ▼
MediaPipe Hands
    │
    ▼
Extraction des landmarks
(2 mains × 21 points × 3 coordonnées)
    │
    ▼
Normalisation
    │
    ▼
Modèle LSTM
    │
    ▼
Prédiction
    │
    ▼
Interface Streamlit
```

---

## Architecture du projet

```text
sign-language-translator/
│
├── data/
│   ├── raw/              # Séquences de landmarks (.npy)
│   ├── processed/        # Prétraitements futurs
│   └── external/         # Vidéos externes optionnelles
│
├── models/
│   └── v1/
│       ├── model.keras
│       └── labels.json
│
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── collect_data.py
│   ├── import_videos.py
│   ├── train.py
│   ├── predict.py
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

Le dataset a été **créé spécifiquement pour ce projet**.

Il est constitué de :

- **6 signes**
- **50 séquences par signe**
- **300 séquences au total**
- **30 frames par séquence**
- **126 caractéristiques par frame**
  - 2 mains
  - 21 landmarks
  - 3 coordonnées (x, y, z)

Les données sont collectées via la webcam, puis converties en séquences de landmarks grâce à **MediaPipe Hands** avant d'être enregistrées au format `.npy`.

---

## Installation

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Utilisation

Toutes les commandes sont exécutées depuis le dossier **src/**.

### 1. Collecte des données

```bash
cd src
python collect_data.py
```

Le script :

- ouvre la webcam ;
- affiche un compte à rebours ;
- capture automatiquement une séquence de **30 frames** ;
- sauvegarde les données au format `.npy`.

Il reprend automatiquement la collecte si celle-ci est interrompue.

---

### Import de vidéos (optionnel)

Déposer les vidéos dans :

```text
data/external/<nom_du_signe>/
```

Puis lancer :

```bash
python import_videos.py
```

---

### 2. Entraîner le modèle

```bash
python train.py
```

Le script :

- charge les données ;
- entraîne un modèle **LSTM empilé** ;
- utilise **EarlyStopping** et **ModelCheckpoint** ;
- sauvegarde le meilleur modèle ;
- génère la matrice de confusion et le rapport de classification.

---

### 3. Reconnaissance en temps réel

```bash
python predict.py
```

Le modèle :

- ouvre la webcam ;
- reconnaît les signes en continu ;
- affiche le mot reconnu ainsi que son niveau de confiance.

---

### 4. Interface Web

```bash
streamlit run app.py
```

L'application est ensuite accessible à l'adresse :

```
http://localhost:8501
```

---

## Choix techniques

### Pourquoi un LSTM ?

Les signes de la LSF sont principalement des **mouvements**.

Un modèle statique apprend uniquement une posture et confond facilement certains gestes similaires.

Le LSTM apprend la **dynamique du mouvement** sur une séquence complète.

---

### Pourquoi MediaPipe Hands ?

MediaPipe fournit les **21 landmarks** de chaque main.

Le modèle apprend donc à partir des coordonnées des mains plutôt que des pixels de l'image.

Cette approche :

- réduit fortement la quantité de données à traiter ;
- accélère l'entraînement ;
- améliore la robustesse.

---

### Normalisation

Les landmarks sont :

- recentrés sur le poignet ;
- mis à l'échelle.

Le système devient ainsi moins sensible :

- à la distance de la caméra ;
- à la position de la main dans l'image.

---

### Taille fixe des données

Chaque frame est représentée par :

- 2 mains
- 21 landmarks
- 3 coordonnées

soit :

**126 caractéristiques**

Les mains absentes sont remplacées par des zéros afin de conserver une taille constante.

---

### Buffer glissant

La prédiction est réalisée sur les **30 dernières frames**, ce qui permet une reconnaissance fluide en temps réel.

---

### Seuil de confiance

Une prédiction n'est affichée que si la probabilité dépasse **70 %**, afin de limiter les faux positifs.

---

## Résultats (V1)

Dataset :

- **300 séquences**
- **50 séquences par signe**

Performances :

- Accuracy globale : **98 %**
- Reconnaissance en temps réel
- Une légère confusion entre **Salut** et **Au revoir** (1 erreur sur 10 dans le jeu de test)

---

## Limites

Cette première version présente plusieurs limites :

- dataset encore limité ;
- un seul utilisateur lors de la collecte ;
- uniquement les mains sont prises en compte ;
- vocabulaire réduit à 6 signes ;
- interface Streamlit adaptée à un usage local.

---

## Roadmap

### Version 1

- Reconnaissance de 6 signes
- Modèle LSTM
- Interface Streamlit
- Traduction en temps réel

### Version 2

- Ajouter de nouveaux signes
- Améliorer la stabilité des prédictions
- Historique des signes reconnus
- Construction de phrases simples

### Version 3

- Reconnaissance de phrases complètes
- Intégration de MediaPipe Holistic
- Expressions faciales
- Déploiement Web

### Version 4

- Modèle Transformer
- API REST
- Application mobile
- Déploiement Cloud

---

## Technologies

- Python
- OpenCV
- MediaPipe Hands
- TensorFlow / Keras
- NumPy
- Streamlit
- Scikit-learn

---

## Licence

Projet réalisé dans le cadre d'un projet de fin de formation.
