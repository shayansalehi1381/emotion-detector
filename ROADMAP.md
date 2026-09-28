# ROADMAP — Real-Time Facial Emotion Recognition (Webcam)

The project's progress reference. Check off `[x]` each completed step. Work proceeds
phase by phase and autonomously.

## Phase 0 — Setup
- [x] Create `.venv` with `py -3.11 -m venv .venv`
- [x] Install PyTorch with CUDA support (torch==2.6.0+cu124, matching the installed driver)
- [x] Install remaining packages (numpy, pandas, matplotlib, scikit-learn, opencv-python, mediapipe, tqdm, seaborn)
- [x] GPU test: `src/check_gpu.py` — confirmed: RTX 3060 Laptop GPU detected
- [x] Folder skeleton: `src/`, `data/`, `checkpoints/`, `results/`, `docs/`
- [x] `.gitignore` (data/*.csv, .venv/, __pycache__/, checkpoints/*.pt, ...)
- [x] `requirements.txt`
- [x] Independent `git init` for this project + first commit
- [x] `docs/PHASE_0_REPORT.md`

## Phase 1 — Data
- [x] Inspect the CSV format (columns, row count, Usage distribution)
- [x] Class distribution chart (sample count per emotion) at `results/class_distribution.png`
- [x] `src/dataset.py`: reconstruct 48x48 images from the pixel string + PyTorch Dataset class
- [x] DataLoader for train/val/test per the Usage column (`get_dataloaders`)
- [x] Sanity check: show a few samples from each class at `results/sample_grid.png`
- [x] `docs/PHASE_1_REPORT.md`

## Phase 2 — Baseline Model
- [x] `src/model.py`: a simple CNN from scratch (683,527 parameters)
- [x] Overfit test on a small batch (`src/overfit_test.py` — succeeded, 100% on the batch)
- [x] `src/train.py` with logging (loss/accuracy per epoch)
- [x] First full training run (30 epochs, best val_acc=56.84% at epoch 11)
- [x] Loss and accuracy chart (train/val) at `results/baseline_curves.png`
- [x] `docs/PHASE_2_REPORT.md`

## Phase 3 — Model Improvements
- [x] Add BatchNorm
- [x] Add Dropout
- [x] Data augmentation (flip, rotation, translation)
- [x] LR scheduler (ReduceLROnPlateau)
- [x] Class weighting (the smoother sqrt version outperformed the raw one)
- [x] Early stopping
- [x] Experiment comparison table at `results/experiments_comparison.csv` + chart
- [x] Save the best checkpoint: `checkpoints/best.pt` (val_acc=60.55%)
- [x] `docs/PHASE_3_REPORT.md`

## Phase 4 — Evaluation
- [x] Final evaluation on PrivateTest (test set) — test accuracy=60.60%
- [x] Confusion matrix at `results/confusion_matrix.png`
- [x] Per-class precision/recall/F1 at `results/classification_report.txt`
- [x] Error analysis with misclassified sample images at `results/error_analysis.png`
- [x] `docs/PHASE_4_REPORT.md`

## Phase 5 — Live Application
- [x] Face detection with OpenCV Haar Cascade on webcam frames (the installed mediapipe version lacked the classic API)
- [x] Preprocessing identical to training time (resize, grayscale, normalize)
- [x] Smoothing predictions across frames (moving average over the last 8 frames)
- [x] Live display: emotion + confidence percentage + FPS on screen
- [x] Clean exit with the `q` key
- [x] `docs/PHASE_5_REPORT.md`

## Phase 6 — Final Documentation
- [x] Professional README.md (project overview, setup, usage, results, limitations)
- [x] `docs/INTERVIEW_NOTES.md` (summary of all concepts for interview prep)
- [x] `docs/DEMO_RECORDING_GUIDE.md` (demo recording guide)

## Current Status
All phases (0-6) are complete. The project is finished.
Summary: val_acc=60.55%, test_acc=60.60% on PrivateTest, final model at checkpoints/best.pt.
