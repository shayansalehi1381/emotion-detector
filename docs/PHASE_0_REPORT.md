# Phase 0 — Setup Report

## What we did and why

The goal of this phase was purely to prepare the environment, with no machine learning
code at all. The reason: if the environment (GPU, packages, folder structure) isn't
right from the start, later phases fill up with errors unrelated to machine learning.

Work completed:

1. **Virtual environment (venv)**: created an isolated Python environment with
   `py -3.11 -m venv .venv`. Why does this matter? Because this project's packages (a
   specific version of PyTorch with CUDA) could conflict with packages installed
   globally on the system.
2. **Install PyTorch with CUDA**: installed `torch==2.6.0+cu124` from PyTorch's official
   index (compatible with the system's GPU driver). This version runs computations on
   the GPU (graphics card) instead of the CPU, which is tens of times faster for
   convolutional neural networks (CNNs).
3. **Install the remaining packages**: pandas/numpy for working with data,
   matplotlib/seaborn for plotting, scikit-learn for evaluation metrics
   (precision/recall/F1), opencv-python and mediapipe for face detection in Phase 5,
   tqdm for progress bars.
4. **GPU test** (`src/check_gpu.py`): ran a real matrix multiplication on the GPU to
   make sure the install isn't just correct "on paper" but actually works.
5. **Folder structure**: `src/` (code), `data/` (dataset), `checkpoints/` (model
   weights), `results/` (charts and tables), `docs/` (reports).
6. **`.gitignore`**: the dataset file (about 300MB) and the virtual environment were
   excluded from git since they're large and reproducible; only code should be in git.
7. **Independent `git init`**: since this folder sits inside a much larger git repo
   (the entire Windows user folder), we created a new, separate git repo just for this
   project so its history doesn't get mixed with unrelated files elsewhere on the system.

## Real Results

```
PyTorch version: 2.6.0+cu124
CUDA available: True
GPU name: NVIDIA GeForce RTX 3060 Laptop GPU
CUDA version (build): 12.4
GPU matmul test OK, result shape: torch.Size([1000, 1000]), device: cuda:0
```

Initial dataset check (`data/fer2013.csv`):

```
Columns: emotion, pixels, Usage
Total samples: 35,887
Training:    28,709 samples
PublicTest:   3,589 samples  (used for validation)
PrivateTest:  3,589 samples  (used for the final test)
```

These numbers exactly match FER2013's standard, well-known split.

## New Concepts (with a simple example)

- **Virtual environment (venv)**: like an "isolated folder" for installing Python
  packages, so installing a specific version of a library for this project doesn't
  break the version installed for other projects.
- **CUDA**: NVIDIA's technology that lets general-purpose code (like matrix
  multiplication) run on the GPU's parallel cores. A GPU has thousands of simple cores
  that are much faster than a few powerful CPU cores for the matrix operations of
  neural networks (which are really just matrix multiplication and addition).
- **train/validation/test split**: three separate portions of the data. Train is used
  to learn the model, validation is used to tune settings (like picking the best
  epoch) without looking at the final data, and test is used only once at the end to
  report the model's real performance on completely unseen data.

## Likely Interview Questions

**1. Why did you use a GPU instead of a CPU?**
Because a CNN involves millions of matrix multiplications and additions that can run
in parallel. GPUs are designed for exactly this and cut training from hours down to
minutes.

**2. Why didn't you commit the dataset to git?**
The dataset file is about 300MB. Committing large files to git bloats the repo's
history, and the best practice is to keep them separate (or use a tool like Git LFS).

**3. Why do you need three splits — train/validation/test — isn't train/test enough?**
If we only had train/test and checked test repeatedly to pick the best model, we'd
effectively be indirectly overfitting the model to test. Validation is an "intermediate
test field" that lets us make decisions (like when to stop training) without
contaminating the final test.

**4. Why did you create a virtual environment (venv)?**
To isolate this project's dependencies from packages installed globally on the system,
and for reproducibility (anyone can build the same environment with requirements.txt).

**5. How is the CUDA version you installed compatible with the system's GPU driver?**
NVIDIA drivers support older CUDA Toolkit versions too (backward compatible). The
system's driver supported a much newer CUDA version, so cu124 ran on it without issue;
we confirmed this by actually running `check_gpu.py`, not just by assuming it.

## Status
Phase 0 complete. Ready for Phase 1 (data review and preparation).
