import cv2
from deepface import DeepFace

# Carregue o modelo de detecção de emoção pré-treinadox
model = DeepFace.build_model("Emotion")

# Definir lista de emoção
emotion_labels = ['nervoso', 'desgosto', 'medo', 'feliz', 'triste', 'surpredo', 'neutro']

# Carregar classificador de cascata facial
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Iniciar captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    # Captura quadro a quadro
    ret, frame = cap.read()

    # Converter quadro em escala de cinza
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos no quadro
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extraia o rosto: ROI (Region of Interest)
        face_roi = gray_frame[y:y + h, x:x + w]

        # Redimensione a face ROI para corresponder à forma de entrada do modelo
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalizar a imagem do rosto redimensionada
        normalized_face = resized_face / 255.0

        # Remodele a imagem para corresponder à forma de entrada do modelo
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)

        # busca emoções usando o modelo pré-treinado
        preds = model.predict(reshaped_face)[0]

        emotion_idx = preds.argmax()
        
        #Pega o index e busca na lista de emoções
        emotion = emotion_labels[emotion_idx]

        # Desenhe um retângulo ao redor do rosto e marque com a emoção prevista
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 15, 65), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 15, 65), 2)

    # Exibir o quadro resultante com o TITULO e frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Solte a captura e feche todas as janelas
cap.release()
cv2.destroyAllWindows()
