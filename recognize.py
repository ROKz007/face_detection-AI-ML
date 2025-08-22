import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load Haar Cascade reliably
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load trained CNN model
model = load_model("final_model.h5")

# Load label encoder (so no need for hardcoded labels)
encoder = joblib.load("label_encoder.pkl")

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1, 100, 100, 1)
    img = img / 255.0
    return img

# Open laptop webcam
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture frame from webcam. Exiting...")
        break

    faces = classifier.detectMultiScale(frame, 1.5, 5)

    for x, y, w, h in faces:
        face = frame[y:y + h, x:x + w]

        # Predict label
        preds = model.predict(preprocess(face))
        pred_idx = np.argmax(preds)
        label = encoder.classes_[pred_idx]   # Get actual name

        # Draw bounding box & label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
