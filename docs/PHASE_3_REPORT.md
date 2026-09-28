# Phase 3 — Model Improvement Report

## What we did and why

The goal of this phase was to close the overfitting gap we saw in Phase 2
(train_acc=97% vs. val_acc=57%). This phase was the most instructive part of the
project because **the initial result went against expectations**, and we had to
diagnose the cause and fix it — exactly as happens in real machine learning work.

### Techniques Added

1. **BatchNorm** (`EmotionCNNImproved` in `src/model.py`): a `BatchNorm2d` was added
   after every Conv layer. It normalizes each layer's output (zero mean, unit
   variance) so the next layer's input distribution shifts less during training,
   making training more stable/faster.
2. **Dropout**: added before both fully-connected layers (with rates 0.5 and 0.3). At
   every training step, a fraction of neurons are randomly turned off so the model
   doesn't become dependent on any one specific path (a kind of implicit ensemble).
3. **Data Augmentation** (`get_train_transform` in `src/dataset.py`): applied only to
   train — random horizontal flip, rotation up to 10 degrees, and translation up to
   10% of the image. val/test are left unchanged (they must be deterministic and reproducible).
4. **Class weighting** (`get_class_weights`): since Disgust has about 16x fewer samples
   than Happy, inverse-frequency weights were given to `CrossEntropyLoss` so errors on
   minority classes affect the loss more.
5. **LR Scheduler** (`ReduceLROnPlateau`): if val_loss doesn't improve for several
   epochs, the learning rate is halved — helping the model take finer steps toward a
   minimum near the end of training.
6. **Early stopping**: if val_accuracy doesn't improve for `patience` consecutive
   epochs, training stops early (avoiding wasted time and further overfitting).

### The Real Story of This Phase: An Unexpected Result and Diagnosing It

We ran the experiments back to back (`src/experiments.py`) and got these results (val accuracy):

| Experiment | val accuracy | best epoch |
|---|---|---|
| baseline (Phase 2) | 56.84% | 11 |
| +BatchNorm+Dropout | 58.04% | 30 |
| +Augmentation | 55.50% | 36 |
| +ClassWeight(raw)+Scheduler | **48.34%** | 48 |

The fourth experiment (everything combined), instead of improving, was **worse than
baseline**! Before jumping to a hasty conclusion, we looked at train_acc: at that
experiment's best epoch, train_acc was only 38.5% — meaning, unlike Phase 2 (where the
problem was overfitting), here the model wasn't even doing well on the train data. This
is a sign of **underfitting**, not overfitting.

**Hypothesis:** the raw weight for the Disgust class (about 9.4x) was too aggressive.
Combined at the same time with augmentation (which itself makes the model's job
harder) and a fast learning-rate decay (`patience=3`), these combined constraints
exceeded the model's capacity and epoch budget.

**Diagnostic experiment** (`src/final_v2.py`): to test the hypothesis, we made the
class weight milder (the square root of the raw weight, so Disgust went from 9.4x to
~3.07x), increased the scheduler's patience from 3 to 5 epochs, and gave it more
epochs/patience (70 epochs, early-stop patience=15).

**Result:** val accuracy reached **60.55%** (at epoch 62) — the best result in the
entire project, beating both the baseline and every other experiment. The hypothesis
was confirmed.

### Final Comparison Table

| Experiment | val accuracy | best epoch | epochs run |
|---|---|---|---|
| baseline (Phase 2) | 56.84% | 11 | 30 |
| +BatchNorm+Dropout | 58.04% | 30 | 30 |
| +Augmentation | 55.50% | 36 | 40 |
| +ClassWeight(raw)+Scheduler | 48.34% | 48 | 58 (early stop) |
| **+ClassWeight(smooth/sqrt)+Scheduler (final)** | **60.55%** | **62** | 70 |

Bar chart: `results/experiments_comparison.png`
Loss/accuracy chart for the final model: `results/final_v2_curves.png` — unlike the
Phase 2 baseline, here val_accuracy is always higher than train_accuracy (since train
is harder due to augmentation), and neither has started diverging from the other yet —
meaning the model hasn't overfit yet and might improve slightly further with more
epochs (a note for future work).

**Final model chosen for Phase 4: `checkpoints/best.pt`** (copy of `final_v2.pt`,
val_accuracy=60.55%)

## New Concepts (with a simple example)

- **BatchNorm**: imagine every layer working with inputs whose scale keeps shifting
  (because earlier layers are also still learning) — this makes learning unstable.
  BatchNorm normalizes each layer's input within every batch so this "moving target"
  becomes more stable.
- **Dropout**: like randomly benching a few team members from practice each time so
  the team doesn't learn to rely on just one specific player; during the final test
  (inference) everyone is present, and the model performs stronger and generalizes
  better.
- **Underfitting vs. Overfitting**: overfitting means the model does very well on train
  and poorly on val (memorizing instead of learning). Underfitting is the opposite: the
  model doesn't even do well on train, usually because regularization constraints are
  too strong or the model/training time isn't sufficient. Telling these apart is done
  by comparing train_acc and val_acc, not by looking at val_acc alone.
- **Why the square root (sqrt) of the class weight?** The raw weight (a direct
  frequency ratio) can become excessively large for very rare classes and dominate the
  gradient. Taking the square root compresses this difference (9.4x raw vs. 3.07x after
  the square root) — a balance between "paying more attention to the rare class" and
  "not destabilizing training."

## Likely Interview Questions

**1. Why did adding class weighting initially make the result worse?**
Because the raw weight (~9.4x for Disgust) combined with augmentation and a fast LR
decay, causing the model to fail to learn well even on train (train_acc only ~38%) —
that's underfitting, not a generalization problem. By softening the weight (square
root) and giving it more room (more epochs/patience), the problem was solved and we
reached the project's best result (60.55%).

**2. How did you know it was underfitting and not a bug in the code?**
I checked train_accuracy, not just val_accuracy. If the model had a bug, it would
typically fail even on a small batch (which we'd already confirmed in Phase 2's
`overfit_test.py`). Here, train_acc was rising slowly and sensibly, just too slowly and
insufficiently — a sign of excessive regularization, not a bug.

**3. Why did you only use flip/rotation/translation for augmentation, not something
stronger like cutout or mixup?**
The images are only 48x48 pixels; aggressive augmentation could destroy important
facial information (like mouth or eye shape, which distinguishes between classes).
Mild geometric augmentations simulate real variation (camera angle, face orientation)
without damaging the image's emotional signal.

**4. The final model still hasn't overfit; why did you stop training at 70 epochs?**
Given this phase's time budget, 70 epochs (roughly 30-40 minutes on this GPU) was a
reasonable stopping point with a good result. The chart shows the model was still
improving gradually; longer training (or a larger early-stopping patience) could have
improved the result a bit more — I recorded this as a known limitation in the report,
rather than pretending the model had reached its ceiling.

**5. How does the 60.55% number compare to the realistic ~65-70% ceiling on FER2013?**
This number is in a reasonable range for a relatively simple CNN (without more complex
architectures like ResNet or a multi-model ensemble), though it hasn't reached the
upper end of that range. The remaining gap is usually closed with a deeper
architecture, longer training, or a multi-model ensemble — outside the scope of this
project (whose goal was learning the full process, not chasing a state-of-the-art record).

## Status
Phase 3 complete. Best checkpoint: `checkpoints/best.pt` (val_accuracy=60.55%).
Ready for Phase 4 (final evaluation on PrivateTest).
