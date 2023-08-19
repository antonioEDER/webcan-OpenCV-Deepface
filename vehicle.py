import cv2

# Iniciar captura de vídeo
cap = cv2.VideoCapture('vehicle.mp4')
# cap = cv2.VideoCapture(0)

# Os classificadores XML treinados descrevem alguns recursos de algum objeto que queremos detectar
car_cascade = cv2.CascadeClassifier('vehicle.xml')

# O loop é executado se a captura foi inicializada
while True:
    ret, frames = cap.read()

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Real-time Detection', frames )
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# De-allocate any associated memory usage
cv2.destroyAllWindows()