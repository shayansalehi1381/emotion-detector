# Interview Prep Notes

This file is a summary of the whole project so you can review it in a few minutes
before an interview. Full details for each section are in the phase-by-phase reports
(`docs/PHASE_*_REPORT.md`).

## Project Summary in One Paragraph

I trained a CNN from scratch (no pretrained model) on the FER2013 dataset (35,887
48x48 grayscale images, 7 emotion classes). First I built a simple baseline that
overfit (train_acc=97%, val_acc=57%). By adding BatchNorm, Dropout, data augmentation,
class weighting, an LR scheduler, and early stopping, I reached val_accuracy=60.55%.
Along the way, one experiment (raw class weighting + augmentation together) gave a
worse result, which I diagnosed as underfitting by looking at train_accuracy (which
had stayed low), and fixed by softening the class weight (square root instead of the
raw ratio). Final evaluation on PrivateTest (completely untouched until that point)
confirmed 60.60% accuracy. Finally, I combined the model with OpenCV's Haar Cascade for
face detection to build a live webcam app showing emotion + confidence + FPS.

## Key Numbers (memorize these)

| Item | Number |
|---|---|
| Total samples | 35,887 |
| Training / PublicTest / PrivateTest | 28,709 / 3,589 / 3,589 |
| Number of classes | 7 |
| Imbalance ratio (Happy / Disgust) | ~16x |
| Final model parameters | ~684K |
| Baseline val_acc (no regularization) | 56.84% |
| Final model val_acc | 60.55% |
| Final test_acc (PrivateTest) | 60.60% |
| Best class (F1) | Happy = 0.838 |
| Weakest class (F1) | Fear = 0.273 |
| Human accuracy on FER2013 (Goodfellow et al., 2013) | 68±5% |
| Best published single model (no ensemble) | ~73.28% |
| Best reported heavy ensemble | ~75.8% |

> Based on the official paper introducing the FER2013 dataset (Goodfellow et al.,
> 2013), human accuracy on this dataset is 68±5% (other estimates in the research
> literature cite 65 to 68%). The best published models in the world (single model, no
> ensemble) have reached about 73%, and with a heavy multi-model ensemble, the best
> reported result is about 75.8%. That means even professional research teams with far
> more computational resources than this project have never reached accuracy close to
> 90% — this ceiling comes not from weak models but from the dataset's inherent
> difficulty (low quality of posed facial expressions in the images, and labeling done
> via web image search).
>
> Sources:
> - Goodfellow et al., "Challenges in Representation Learning: A report on three
>   machine learning contests" (arxiv.org/pdf/1307.0414) — human accuracy 68±5%
> - "Facial Emotion Recognition: State of the Art Performance on FER2013"
>   (arxiv.org/pdf/2105.03588) — best single model 73.28%
> - "Facial Expression Recognition with Deep Learning"
>   (cs230.stanford.edu/projects_winter_2020/reports/32610274.pdf) — best reported
>   ensemble 75.8%

## Key Concepts You Should Be Able to Explain

1. **Train/Val/Test split and why three sets**: val is for tuning the model, test is
   used only once at the end — preventing indirect leakage of test information into
   the decision-making process.
2. **Overfitting vs. Underfitting**: diagnosed by comparing train_acc and val_acc, not
   by looking at just one number. Overfitting = large gap (train high, val low).
   Underfitting = both low.
3. **BatchNorm and Dropout**: BatchNorm increases training stability (normalizing each
   layer's input); Dropout prevents excessive reliance on specific neurons by randomly
   turning them off.
4. **Class imbalance and class weighting**: inverse-frequency weighting in the loss; a
   weight that's too aggressive can cause underfitting (a real experience from this
   project) — a square-rooted weight is milder.
5. **Data augmentation**: artificial variety (flip/rotation/translation) to reduce
   overfitting; it must be mild, otherwise it destroys the image's main signal (here,
   facial expression).
6. **Precision/Recall/F1 vs. Accuracy**: on imbalanced data, accuracy can be
   misleading; per-class reporting is needed.
7. **Confusion Matrix**: shows *where* the model makes mistakes, not just *how much*.
8. **Preprocessing consistency between train and inference**: one of the most common practical bugs.

## The Phase 3 Diagnostic Story (Best Likely Interview Question)

Make sure you have this section ready — it demonstrates genuine debugging ability, not
just following instructions:

> "When I added all the techniques together (augmentation + class weight +
> scheduler), the result got worse than baseline (48% instead of 57%). Before
> concluding these techniques were useless, I checked train_accuracy and saw it had
> also stayed low (~38%) — meaning the problem wasn't overfitting, but the opposite:
> the model wasn't even learning well on train (underfitting). I hypothesized that the
> raw weight for the Disgust class (9.4x) had been too aggressive and, combined with
> augmentation, over-constrained the model. By softening the weight (square root
> instead of the raw ratio) and giving it more epochs, the result reached the best
> number in the whole project (60.55%)."

## Frequently Asked Questions and Short Answers

**Why didn't you use a pretrained model?**
It was a project constraint — the goal was to learn the full process of training a CNN
from scratch, not just fine-tuning a ready-made model.

**Why FER2013 and not another dataset?**
It's a standard, well-known dataset in emotion recognition research, has a manageable
size, and comes ready in CSV format with an official train/val/test split.

**Is your final accuracy good?**
Our model's 60.60% number, given that it was trained from scratch (no pretrained
model) on a relatively simple CNN architecture, is in a reasonable range relative to
human accuracy (68%), though it hasn't reached the ceiling of more complex models
(73-76%). The best class (Happy) performs excellently with F1=0.838; the weakest
(Fear) was more challenging due to inherent visual similarity to other classes and
some label noise in the dataset itself.

**What would you do with more time?**
A deeper architecture (e.g. with residual connections), stronger augmentation
targeted at hard classes (Fear), focal loss instead of/alongside class weighting, and
longer training (the final model's chart shows it was still improving gradually, not
yet fully converged).

**Why did you use Haar Cascade instead of a neural network for face detection?**
The installed mediapipe version didn't have its classic API (it was a much newer
version) and needed a separately downloaded model; Haar Cascade, with no extra
dependency, was good enough for this demo.
