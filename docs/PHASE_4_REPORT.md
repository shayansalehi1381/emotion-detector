# Phase 4 — Final Evaluation Report

## What we did and why

Up to this point, every decision (architecture choice, hyperparameters, best epoch)
had been made based only on train/val. PrivateTest had remained completely untouched
until this moment — not a single sample from it had been seen during training or model
selection. This phase ran on PrivateTest **exactly once** to get the most honest
possible estimate of the model's performance on truly new data. (`src/evaluate.py`)

## Real Results

```
Number of PrivateTest samples: 3589
Test accuracy: 60.60%
```

This number is nearly identical to Phase 3's val_accuracy (60.55%) — the very small
gap (0.05 percentage points) shows the model didn't overfit to the val set either, and
this number is a reliable estimate of the model's real performance.

### Per-class Report (precision/recall/f1)

| Class | precision | recall | f1-score | support |
|---|---|---|---|---|
| Angry    | 0.580 | 0.470 | 0.520 | 491 |
| Disgust  | 0.431 | 0.509 | 0.467 | 55  |
| Fear     | 0.446 | 0.197 | **0.273** (weakest) | 528 |
| Happy    | 0.822 | 0.854 | **0.838** (best) | 879 |
| Sad      | 0.419 | 0.540 | 0.472 | 594 |
| Surprise | 0.723 | 0.767 | 0.744 | 416 |
| Neutral  | 0.546 | 0.673 | 0.603 | 626 |
| **macro avg** | 0.567 | 0.573 | 0.559 | 3589 |
| **weighted avg** | 0.601 | 0.606 | 0.593 | 3589 |

Interesting point: despite Disgust having the fewest samples (55), after class
weighting it's no longer the worst class (recall=0.509, even better than Angry). But
**Fear**, despite having a relatively large 528 samples, became the weakest class — so
the problem wasn't purely "lack of data"; Fear is inherently very visually similar to
other classes (see the interview section below).

### Confusion Matrix (`results/confusion_matrix.png`)

The biggest error patterns:
- **Fear → Sad** (160 cases), **Fear → Surprise** (83), and **Fear → Neutral** (74):
  Fear in FER2013 is usually accompanied by a frown/wide eyes, which is visually very
  similar to Sad, Surprise, and even Neutral.
- **Sad → Neutral** (152 cases): the difference between Sad and Neutral in 48x48
  black-and-white images can be very subtle (a slight downturn at the corner of the mouth).
- **Angry → Sad** (99 cases): both can involve a frown.
- **Happy** is almost never confused with a negative class (a smile is a very clear
  signal) — precisely why it has the best F1.

### Error Analysis (`results/error_analysis.png`)

We reviewed the 16 samples the model mispredicted with the highest confidence (up to
100%). Key point: **a good chunk of these "errors" are likely the dataset's own
labeling errors**, not genuine model failures:

- Several images labeled Surprise are actually clearly smiling faces (mouth open from
  laughing, not shock) — the model predicted Happy for them with 99-100% confidence,
  which seems more visually sensible. Likewise, one sample labeled Fear is actually a
  clear smile, and the model predicted Happy for it with 100% confidence.
- Watermarks from stock photo companies (like "shutterstock") are visible on several
  images — a sign that this dataset was collected automatically from the internet, and
  label quality isn't always 100% reliable (a known limitation of FER2013 in the research literature).

## New Concepts (with a simple example)

- **Precision vs. Recall**: for the Fear class, precision=0.446 means when the model
  says "Fear," it's right 44.6% of the time. recall=0.197 means out of all actual Fear
  samples, the model only found 19.7% of them (attributing the rest to other classes).
  So the model is both cautious about Fear (rarely says it) and not very confident when
  it does.
- **Macro average vs. Weighted average**: the macro average is the simple mean of F1
  across the seven classes (every class weighted equally, even Disgust with 55
  samples). The weighted average weights by each class's sample count (closer to
  overall accuracy). The gap between the two (0.559 vs. 0.593) shows the model performs
  somewhat worse on minority classes than the weighted average suggests.
- **Confusion Matrix**: a table showing which class the model mistook for which other
  class. The main diagonal = correct predictions. Off-diagonal = error patterns. This
  table tells us far more than a single accuracy number about *where* the model is weak.

## Likely Interview Questions

**1. Why did you only run on PrivateTest once, instead of multiple times to optimize?**
If I ran on test multiple times and made decisions based on the result (e.g. changing
hyperparameters), I would effectively be turning test into a second validation set, and
the final number would no longer be an unbiased estimate of performance on truly new
data. This is exactly why we kept val/test separate from Phase 1 onward.

**2. Why did Fear become the weakest class even though it wasn't the smallest class?**
Because the problem isn't just sample count; visual similarity between classes matters
too. Fear in 48x48 black-and-white images overlaps visually a lot with Sad, Surprise,
and Neutral (e.g. raised eyebrows appear in both Fear and Surprise). The confusion
matrix shows this directly: most of Fear's errors go toward exactly these three classes.

**3. Does this mean the model is bad?**
Not necessarily — some errors stem from the dataset itself (the suspicious/noisy labels
we saw in error analysis), and some are inherent to the problem (even a human sometimes
can't tell Fear from Surprise in a single still black-and-white 48x48 photo). The 60.6%
number should be read alongside these limitations, not as "the model performs badly."

**4. Why did you use classification_report instead of just accuracy?**
Because overall accuracy can be misleading when data is imbalanced. For example, if the
model always predicted Happy, it would still be right about 24% of the time (Happy's
proportion in test). Per-class precision/recall/f1 shows how the model actually
performs on each class, not just an overall average.

**5. How could you have improved Fear?**
A few options: stronger augmentation specific to hard classes, separate weighting (not
just based on frequency but on class difficulty), a deeper architecture with more
capacity to learn subtle distinctions, or using focal loss, which automatically focuses
more on hard samples. These stayed outside this project's time budget but are clear
directions for future work.

## Status
Phase 4 complete. Final model: `checkpoints/best.pt`, test accuracy=60.60% on PrivateTest.
Ready for Phase 5 (live webcam application).
