# Real-Time Facial Emotion Recognition (Webcam)

A CNN trained **from scratch** (no pretrained model) on the FER2013 dataset that
recognizes facial emotion from a live webcam feed and displays a confidence score.

This project was built to learn the full process of a deep learning project end to
end: from reading raw data, through training and debugging, to honest evaluation and
shipping the model as a real application.

## Final Results

| Metric | Value |
|---|---|
| Best val accuracy (PublicTest) | **60.55%** |
| Final test accuracy (PrivateTest, evaluated only once) | **60.60%** |
| Best class (F1) | Happy — 0.838 |
| Weakest class (F1) | Fear — 0.273 (high visual similarity to Sad/Surprise) |

Full details, including one experiment that initially performed worse and how it was
diagnosed and fixed, are in [docs/PHASE_3_REPORT.md](docs/PHASE_3_REPORT.md) and
[docs/PHASE_4_REPORT.md](docs/PHASE_4_REPORT.md).

> This number is reported deliberately without exaggeration. The realistic accuracy
> ceiling on FER2013 with a simple CNN is roughly 65-70%; reaching higher numbers
> usually requires deeper architectures or a multi-model ensemble.

## Architecture and Dataset

- **Dataset**: [FER2013](https://www.kaggle.com/datasets/deadskull7/fer2013) — 35,887
  48x48 grayscale images across 7 emotion classes (Angry, Disgust, Fear, Happy, Sad,
  Surprise, Neutral), with the official Training/PublicTest/PrivateTest split.
- **Model**: `EmotionCNNImproved` — three Conv+BatchNorm+ReLU+MaxPool blocks, plus
  Dropout before the fully-connected layers. About 684K parameters.
- **Improvement techniques**: data augmentation (flip/rotation/translation), class
  weighting with a smoothed (sqrt) weight for class imbalance, an LR scheduler, and
  early stopping.
- **Live face detection**: OpenCV Haar Cascade (not part of the trained model — used
  only to locate the face in the webcam frame before feeding it to the model).

## Project Structure

```
src/
  dataset.py          Dataset class, DataLoader, augmentation, class weights
  model.py             EmotionCNN (baseline) and EmotionCNNImproved architectures
  train.py             Generic training loop + logging + checkpointing
  experiments.py        Phase 3 experiments (BatchNorm+Dropout, augmentation, ...)
  final_v2.py            Final, winning experiment from Phase 3
  evaluate.py            Final evaluation on PrivateTest
  live_demo.py            Live webcam application
  explore_data.py, check_gpu.py, check_samples.py, overfit_test.py,
  plot_training.py, compare_experiments.py     Helper/debug tools
data/fer2013.csv       Raw dataset (not in git, you must download it yourself)
checkpoints/            Saved model weights (not in git)
results/                Plots, tables, evaluation reports
models/                 Haar Cascade file for face detection
docs/                   Per-phase reports + interview notes
```

## Setup

```bash
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the dataset from Kaggle —
[deadskull7/fer2013](https://www.kaggle.com/datasets/deadskull7/fer2013) — and place
the CSV at `data/fer2013.csv`.

> Note: the default Windows console does not support UTF-8. Run every script with
> `PYTHONIOENCODING=utf-8` so console output prints correctly.

## Usage

```bash
# Check GPU availability
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe src/check_gpu.py

# Train the improved model with the best settings (Phase 3 result)
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe src/final_v2.py

# Final evaluation on PrivateTest
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe src/evaluate.py

# Live webcam app (press q to quit)
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe src/live_demo.py
```

## Known Limitations

- Accuracy on the Fear class is weak (high visual similarity to Sad/Surprise/Neutral
  in small grayscale images — details in [docs/PHASE_4_REPORT.md](docs/PHASE_4_REPORT.md)).
- FER2013 itself has some known label noise (a few samples found during error analysis
  had labels that didn't match the actual image content).
- Haar Cascade face detection performs worse than deep-learning-based methods under
  poor lighting or non-frontal face angles.
- The model handles one face at a time (the largest face detected).

## Full Documentation Per Phase

- [docs/PHASE_0_REPORT.md](docs/PHASE_0_REPORT.md) — Environment setup
- [docs/PHASE_1_REPORT.md](docs/PHASE_1_REPORT.md) — Data
- [docs/PHASE_2_REPORT.md](docs/PHASE_2_REPORT.md) — Baseline model
- [docs/PHASE_3_REPORT.md](docs/PHASE_3_REPORT.md) — Model improvements (includes an
  unexpected result and how it was diagnosed/fixed)
- [docs/PHASE_4_REPORT.md](docs/PHASE_4_REPORT.md) — Final evaluation
- [docs/PHASE_5_REPORT.md](docs/PHASE_5_REPORT.md) — Live application
- [docs/INTERVIEW_NOTES.md](docs/INTERVIEW_NOTES.md) — Interview prep summary
- [docs/DEMO_RECORDING_GUIDE.md](docs/DEMO_RECORDING_GUIDE.md) — Demo recording guide
- [ROADMAP.md](ROADMAP.md) — Phase-by-phase project progress
