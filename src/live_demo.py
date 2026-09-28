"""
برنامه‌ی زنده‌ی فاز ۵: از وب‌کم فریم می‌خواند، چهره را با Haar Cascade (OpenCV) پیدا
می‌کند، همان preprocessing زمان train (خاکستری، ۴۸×۴۸، نرمال [0,1]) را روی چهره اعمال
می‌کند، و حالت احساسی + درصد اطمینان + FPS را روی تصویر نشان می‌دهد.
پیش‌بینی‌ها با میانگین متحرک بین فریم‌ها هموار می‌شوند تا لرزش نمایش کم شود.
خروج با کلید q.
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
SMOOTH_WINDOW = 8  # تعداد فریم‌هایی که برای میانگین متحرک نگه می‌داریم

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"استفاده از دستگاه: {device}")

model = EmotionCNNImproved().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()
print(f"مدل از {CHECKPOINT_PATH} بارگذاری شد.")

face_detector = cv2.CascadeClassifier(CASCADE_PATH)
if face_detector.empty():
    raise RuntimeError(f"فایل cascade در {CASCADE_PATH} پیدا نشد یا خراب است.")

# صف پیش‌بینی‌های اخیر برای میانگین متحرک (هموارسازی)
prob_history = deque(maxlen=SMOOTH_WINDOW)


def preprocess_face(gray_face):
    """همان preprocessing زمان train: resize به 48x48، نرمال به [0,1]، تبدیل به تنسور."""
    resized = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE))
    tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    return tensor.to(device)


def predict_emotion(gray_face):
    """پیش‌بینی خام مدل روی یک چهره، بدون هموارسازی."""
    tensor = preprocess_face(gray_face)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("وب‌کم پیدا نشد. مطمئن شوید دوربین متصل و در دسترس است.")

    print("برنامه‌ی زنده شروع شد. برای خروج کلید q را بزنید.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("خواندن فریم از وب‌کم شکست خورد.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) > 0:
            # بزرگ‌ترین چهره را به‌عنوان چهره‌ی اصلی در نظر می‌گیریم
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
            # cv2.putText فقط از فونت‌های Hershey (لاتین) پشتیبانی می‌کند، پس متن روی تصویر انگلیسی است
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
    print("برنامه بسته شد.")


if __name__ == "__main__":
    main()
