import cv2
from ultralytics import YOLO
from keras.models import load_model
import numpy as np
import smtplib
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv

load_dotenv()

TO_EMAIL = os.getenv("TO_EMAIL")
FROM_EMAIL = os.getenv("FROM_EMAIL")
FROM_PASSWORD = os.getenv("FROM_PASSWORD")

def send_email_alert(emotion, to_email, from_email, from_password):
    subject = f"Alerte Emotion détectée : {emotion}"
    body = f"L'émotion détectée est : {emotion}"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, from_password)
            server.sendmail(from_email, to_email, msg.as_string())
        print(f"Email envoyé pour l'émotion : {emotion}")
    except Exception as e:
        print(f"Erreur lors de l'envoi de l'email : {e}")

class EmotionDetector:
    def __init__(self, model_path="emotion_model.h5"):
        self.model = load_model(model_path)
        self.emotions = ["Colère", "Dégoût", "Peur", "Heureux", "Triste", "Surpris", "Neutre"]

    def predict_emotion(self, face_img):
        face_img = cv2.resize(face_img, (48, 48))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = face_img.astype("float32") / 255.0
        face_img = np.expand_dims(face_img, axis=[0, -1])
        preds = self.model.predict(face_img)
        emotion_idx = np.argmax(preds)
        return self.emotions[emotion_idx]

class YoloWebcamDetector:
    def __init__(self, yolo_model_path="yolov8n.pt", emotion_model_path="emotion_model.h5", cam_index=0):
        self.model = YOLO(yolo_model_path)
        self.emotion_detector = EmotionDetector(emotion_model_path)
        self.cap = cv2.VideoCapture(cam_index)
        self.to_email = TO_EMAIL
        self.from_email = FROM_EMAIL
        self.from_password = FROM_PASSWORD
        if not all([self.to_email, self.from_email, self.from_password]):
            raise ValueError("Veuillez définir TO_EMAIL, FROM_EMAIL et FROM_PASSWORD dans le fichier .env")

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.predict(source=rgb_frame, show=False, verbose=False)
        annotated_frame = frame.copy()
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            emotion = self.emotion_detector.predict_emotion(face)
            if emotion in ["Triste", "Heureux"]:
                send_email_alert(emotion, self.to_email, self.from_email, self.from_password)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(annotated_frame, emotion, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        return annotated_frame

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            annotated_frame = self.process_frame(frame)
            cv2.imshow("YOLOv8 + Emotion Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cap.release()
        cv2.destroyAllWindows()