# Phase 2 — Baseline Model Report (Baseline CNN)

## What we did and why

The goal of this phase is to have an honest baseline: the simplest reasonable CNN,
with no regularization tricks at all, so that in Phase 3 we can measure the real
effect of each improvement by comparing against this baseline.

1. **`src/model.py` — the `EmotionCNN` architecture**: three `Conv -> ReLU -> MaxPool`
   blocks that gradually increase channel depth from 1 (grayscale) to 32, then 64,
   then 128, halving spatial dimensions each time (48→24→12→6). Then `Flatten` and two
   fully-connected layers to reach 7 classes. 683,527 parameters total — a small model
   that comfortably fits in 6GB VRAM.
2. **Overfit test on a small batch** (`src/overfit_test.py`): ran 32 fixed samples
   through 200 forward/backward passes. If the model can't memorize this small batch,
   something in the pipeline is buggy (e.g. an incorrect loss computation, or gradients
   that aren't flowing). Result: loss approached zero and accuracy on that batch
   reached 100%, meaning the model, loss, and optimizer are working correctly.
3. **`src/train.py`**: a generic training loop with a CSV log per epoch (train/val loss
   and accuracy) and automatic saving of the best checkpoint based on val accuracy.
4. **First training run (full baseline)**: 30 epochs, batch size=128,
   optimizer=Adam(lr=0.001), without BatchNorm/Dropout/Augmentation.
5. **Chart** (`results/baseline_curves.png`): loss and accuracy for train and val.

## Real Results

```
Model parameter count: 683,527
```

Training progress (summary; full table in results/train_log_baseline.csv):

| epoch | train_acc | val_acc | train_loss | val_loss |
|-------|-----------|---------|------------|----------|
| 1     | 30.7%     | 39.5%   | 1.72       | 1.57     |
| 5     | 54.2%     | 53.2%   | 1.21       | 1.26     |
| 11    | 69.2%     | **56.8%** (best) | 0.84 | 1.22 |
| 20    | 88.4%     | 55.0%   | 0.33       | 1.90     |
| 30    | 97.1%     | 53.1%   | 0.09       | 3.52     |

**Best val accuracy: 56.84% (at epoch 11)** — checkpoint saved to `checkpoints/baseline.pt`.

The chart clearly shows a classic overfitting pattern: train loss keeps dropping
non-stop and train accuracy reaches 97%, but val loss starts rising from around epoch
~5 onward, and val accuracy oscillates around 53-57% with no real improvement after
epoch 11.

This number falls short of the realistic ~65-70% ceiling on FER2013 — exactly as
expected from a simple model without regularization. Phase 3 is specifically about
closing this gap.

## New Concepts (with a simple example)

- **Overfitting**: when a model memorizes the details (even noise) of the train data
  instead of learning general patterns. The telltale sign is exactly what we saw:
  train accuracy climbs while val accuracy stays flat or gets worse. Like a student who
  memorized the answers to one specific practice book instead of actually learning the
  concept — they do poorly on new questions.
- **Overfit test on a small batch**: a standard deep learning debugging technique.
  Before spending time on a full training run (which can take minutes to hours), we
  first confirm the model can at least fully memorize a small batch. If it can't, the
  problem is definitely in the code, not in the difficulty of the problem itself.
- **MaxPool2d**: a layer that keeps only the largest value in each 2x2 window and
  discards the rest. This both reduces image dimensions (faster computation) and makes
  the model somewhat more robust to small shifts of features within the image.

## Likely Interview Questions

**1. Why did the baseline model overfit, and how do you know?**
Because it had no regularization mechanism (BatchNorm, Dropout, augmentation), and
683K parameters isn't excessive for 28,709 train samples, yet the model reached 97% on
train while val stayed around 53-57%. The large gap between train_acc and val_acc
(97% vs. 53%) is precisely the definition of overfitting.

**2. Why did you run the small-batch overfit test before full training?**
Because if I ran full training (30 epochs, several minutes) and got a bad result, I
wouldn't know whether the problem is the architecture/hyperparameters or a simple code
bug (like mislabeled data). The small-batch test rules that out within seconds.

**3. Which loss function did you use, and why?**
`CrossEntropyLoss`, the standard for multi-class classification. It internally applies
softmax to the model's raw output (logits) and then computes the negative
log-likelihood of the correct class, so I don't need to apply softmax myself beforehand.

**4. How did you pick the best epoch instead of just using the last one?**
Because after epoch 11, val accuracy no longer showed real improvement and val loss
started rising (a sign of overfitting). The code automatically saves a checkpoint only
when val accuracy exceeds the previous best, so the final model is from epoch 11, not epoch 30.

**5. Why did you choose Adam over SGD?**
Adam adapts the learning rate per parameter and is generally a safer choice for a quick
start without needing to hand-tune the learning rate precisely; for a first baseline
whose goal is just to observe the pipeline's overall behavior, it was a safer choice
than plain SGD.

## Status
Phase 2 complete. Ready for Phase 3 (model improvement with regularization).
