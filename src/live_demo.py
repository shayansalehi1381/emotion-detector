"""
Phase 5 live application: reads webcam frames, finds the face with Haar Cascade
(OpenCV), applies the same preprocessing as train time (grayscale, 48x48, normalize to
[0,1]) to the face, and shows the emotion + confidence percentage + FPS on screen.
Predictions are smoothed with a moving average across frames to reduce display jitter.
Exit with the q key.
"""

import time
from collections import deque

import cv2
import numpy as np
import torch

from dataset import EMOTION_NAMES
from model import EmotionCNNImproved

CHECKPOINT_PATH = "checkpoints/best.pt"
CASCADE_PATH = "models/haarcascade_frontalface_default.xml"
IMG_SIZE = 48
SMOOTH_WINDOW = 8  # number of frames kept for the moving average

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = EmotionCNNImproved().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()
print(f"Model loaded from {CHECKPOINT_PATH}.")

face_detector = cv2.CascadeClassifier(CASCADE_PATH)
if face_detector.empty():
    raise RuntimeError(f"Cascade file at {CASCADE_PATH} not found or corrupted.")

# queue of recent predictions for the moving average (smoothing)
prob_history = deque(maxlen=SMOOTH_WINDOW)


def preprocess_face(gray_face):
    """Same preprocessing as train time: resize to 48x48, normalize to [0,1], convert to a tensor."""
    resized = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE))
    tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    return tensor.to(device)


def predict_emotion(gray_face):
    """Raw model prediction on one face, without smoothing."""
    tensor = preprocess_face(gray_face)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found. Make sure a camera is connected and available.")

    print("Live app started. Press q to quit.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read a frame from the webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) > 0:
            # treat the largest face as the main face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y + h, x:x + w]

            probs = predict_emotion(face_roi)
            prob_history.append(probs)
            smoothed_probs = np.mean(prob_history, axis=0)

            pred_class = int(np.argmax(smoothed_probs))
            confidence = smoothed_probs[pred_class]
            emotion_name = EMOTION_NAMES[pred_class]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            label = f"{emotion_name}: {confidence*100:.0f}%"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        else:
            # cv2.putText only supports Hershey (Latin) fonts, so on-screen text is in English
            prob_history.clear()
            cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if current_time > prev_time else 0.0
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Emotion Detector (q = quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("App closed.")


if __name__ == "__main__":
    main()
