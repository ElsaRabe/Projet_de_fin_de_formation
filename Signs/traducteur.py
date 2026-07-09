import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialisation du détecteur de mains (détecte 1 main max)
detector = HandDetector(maxHands=1, detectionCon=0.7)

# Capture vidéo de la webcam
cap = cv2.VideoCapture(0)
print("Système activé via CVZone. Appuyez sur 'q' pour quitter.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Erreur de caméra.")
        break

    # Miroir pour que ce soit plus naturel
    frame = cv2.flip(frame, 1)
    
    # Détecter les mains et dessiner le squelette automatiquement
    hands, frame = detector.findHands(frame)
    
    traduction = "En attente d'un signe..."

    if hands:
        # Récupérer la première main détectée
        hand = hands[0]
        
        # cvzone possède une fonction magique qui dit quels doigts sont levés !
        # Renvoie une liste de 5 valeurs (0 pour fermé, 1 pour ouvert)
        # Ordre : [Pouce, Index, Majeur, Annulaire, Auriculaire]
        fingers = detector.fingersUp(hand)
        
        # Exemple de logique de traduction :
        if fingers == [0, 1, 1, 0, 0]:  # Index et Majeur levés
            traduction = "Traduction : Bonjour / Victoire"
        elif fingers == [0, 1, 0, 0, 0]:  # Uniquement l'index levé
            traduction = "1"
        elif fingers == [1, 1, 1, 1, 1]:  # Tous les doigts levés
            traduction = "Traduction : Salut !"
        elif fingers == [0, 1, 1, 1, 1]:  # Tous les doigts levés
            traduction = "4"
        elif fingers == [1, 1, 0, 0, 1]:
            traduction = "Traduction : Je t'aime / Rock"
        elif fingers == [0, 1, 1, 1, 0]:
            traduction = "3"
        else:
            traduction = "Signe non reconnu"

    # Afficher le texte sur l'image
    cv2.putText(frame, traduction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Afficher la fenêtre
    cv2.imshow('Traducteur LSF', frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()