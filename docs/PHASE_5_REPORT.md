# Phase 5 — Live Webcam Application Report

## What we did and why

The goal of this phase is to bring the Phase 3 trained model into a real, usable
application: grab a webcam frame, find the face, and display the emotion + confidence
percentage live. (`src/live_demo.py`)

### Face Detection: Why Haar Cascade Instead of MediaPipe?

The original plan was to use MediaPipe, but the version installed in this project
(`mediapipe==1.0.1`) is a much newer version that no longer has the classic
`mp.solutions.face_detection` API (used in most tutorials) — only `mediapipe.tasks`
(the new API) is available, which requires downloading a separate `.task` model file.
To keep the project simple, and since the roadmap explicitly allowed "MediaPipe or
OpenCV," we used **Haar Cascade in OpenCV** (`cv2.CascadeClassifier`) — a classic,
lightweight face-detection method.

Another technical note: the installed version of `opencv-python` (5.0.0) no longer
ships Haar Cascade files with the package (unlike older versions). With the user's
explicit permission, the `haarcascade_frontalface_default.xml` file (~930KB) was
downloaded from OpenCV's own official GitHub repo (`opencv/opencv`) and saved to `models/`.

### Live Application Pipeline

1. Read a frame from the webcam (`cv2.VideoCapture(0)`)
2. Convert to grayscale and detect faces with `detectMultiScale`
3. If multiple faces are found, pick the largest one (usually the closest to the camera)
4. **Preprocessing exactly like train time**: resize to 48x48, convert to a tensor,
   divide by 255 (normalize to [0,1]) — any mismatch here with train causes a sharp
   drop in accuracy
5. The model (`EmotionCNNImproved` with `checkpoints/best.pt` weights) predicts
6. **Smoothing**: a moving average over the last 8 probability vectors (not just the
   final label) to reduce frame-to-frame jitter (e.g. flickering between Neutral and
   Sad across consecutive frames)
7. Display a box around the face + emotion label + confidence percentage + FPS on screen
8. Clean exit with the `q` key

### Why a Moving Average Over Probability Instead of Over the Label?

If we only voted over the last N labels, information about "how confident it was" would
be lost. Averaging over the probability vector (before argmax) means that if the model
is slightly unsure between two close classes in one frame, the final output is smoother
and more stable, rather than each frame suddenly jumping between two completely
different labels.

## Real Results and Testing Limitation

The script was run and confirmed on this same system:
- The webcam opened successfully and read a real frame (`640x480`)
- Haar Cascade successfully found a real face in front of the camera
  (example: `faces found: [[179 370 96 96]]`)
- The full script ran for several seconds with no error or traceback (memory and CPU
  usage confirmed the inference loop was actually running)

**Honest limitation**: since my execution environment has no live graphical window I
can view directly, **final visual confirmation** (that the box/label/FPS display
correctly on screen, colors are legible, and smoothing genuinely reduces jitter) must
be done by the user themselves by running the command below directly:

```bash
.venv/Scripts/python.exe src/live_demo.py
```

## New Concepts (with a simple example)

- **Haar Cascade**: a classic (pre-deep-learning-era) face detection method based on
  simple light/dark patterns (e.g. the eye region is usually darker than the cheeks).
  Fast and lightweight (suited to real-time on CPU) but less accurate than newer deep
  learning models (like MediaPipe), especially at non-frontal face angles.
- **Moving Average for smoothing**: imagine measuring air temperature every second and
  the number has some jitter (sensor noise). If you display the average of the last 8
  measurements instead of the instantaneous number, the display becomes much smoother
  and more readable — the same principle applied here to the model's seven-class
  probability vector.
- **Preprocessing consistency between train and inference**: one of the most common
  practical bugs in machine learning is when preprocessing at actual use time
  (inference) differs from train time (e.g. one normalizes to [0,1] while the other
  normalizes with a different mean/std). Even a small difference can sharply reduce
  model accuracy, because the model sees a distribution different from what it learned.

## Likely Interview Questions

**1. Why must inference-time preprocessing exactly match train-time preprocessing?**
During training, the model learned to work with inputs from a specific distribution
(here: pixels normalized to [0,1]). If this same normalization isn't applied at
inference (e.g. raw 0-255 pixels are given instead), the input is completely out of the
distribution the model has ever seen, from its point of view, and predictions become
essentially meaningless, even if the model itself was trained excellently.

**2. Why did you only pick the largest face, not all faces?**
For this simple demo app, the assumption is that there's one main user in front of the
camera. The largest face is usually the person closest to the camera (i.e. the main
user). To support multiple people, it would be enough to loop the prediction over every
detected face (the code's structure already makes this change easy).

**3. Why did you use a simple moving average instead of something more sophisticated like a Kalman filter?**
A simple moving average (over the last 8 frames) was sufficient for this use case: it
has negligible computational cost and clearly reduces frame-to-frame jitter. A Kalman
filter is more useful for problems with an underlying motion model (like position
tracking); here we just want to reduce statistical noise, not track a continuous quantity.

**4. If several people are in front of the camera at once, doesn't the smoothing break?**
With the current implementation (a single global `deque` for the largest face), yes —
if the main face changes (e.g. someone else steps forward), the moving average mixes
between the two people for a few frames. The more correct fix: a separate `deque` per
tracked face (face tracking with an ID), which was kept out of scope for this simple phase.

**5. Why did you use Haar Cascade instead of a neural network for face detection?**
Mainly for a practical reason: the mediapipe version installed in this project didn't
have its classic API and required a separately downloaded model. Haar Cascade, with no
extra dependency, fast, and good enough for a demo with normal lighting and a
front-facing face, was completely sufficient.

## Status
Phase 5 complete (automated testing succeeded; final visual confirmation is up to the user).
Ready for Phase 6 (final documentation).
