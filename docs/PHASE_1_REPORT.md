# Phase 1 — Data Report

## What we did and why

Before building any model, we need to be sure we're reading and understanding the data
correctly. A common mistake is jumping straight to the model and only later
discovering the data was loaded wrong — which wastes a lot of time. That's why this
phase focused entirely on inspecting and preparing the data.

1. **Inspect the CSV format** (`src/explore_data.py`): the `data/fer2013.csv` file has
   three columns: `emotion` (a numeric label 0 to 6), `pixels` (a text string of 2304
   space-separated numbers, i.e. a 48x48 grayscale image), and `Usage`
   (Training/PublicTest/PrivateTest).
2. **Class distribution chart** (`results/class_distribution.png`): shows the data is
   heavily imbalanced — the Disgust class has about 436 samples in train while Happy
   has about 7215 (roughly 16x more). We keep this in mind for Phase 3 (class weighting).
3. **`src/dataset.py`**: the `FER2013Dataset` class, a subclass of
   `torch.utils.data.Dataset`. Its main work happens in `__getitem__`: it splits the
   pixel string, converts it into a 48x48 numpy array, then converts that to a float
   tensor normalized to [0,1]. The train/val/test split is taken directly from the
   dataset's own Usage column (not a manual split) so results are comparable to this
   dataset's standard benchmarks.
4. **DataLoader**: the `get_dataloaders()` function builds three DataLoaders. Only
   train is shuffled, since val/test are only for evaluation and their order doesn't matter.
5. **Sanity check** (`src/check_samples.py` → `results/sample_grid.png`): displayed 5
   samples from each class. The images are clear and the labels match the image
   content sensibly (e.g. Happy samples genuinely show smiles).

## Real Results

```
DataFrame shape: (35887, 3)
Columns: ['emotion', 'pixels', 'Usage']

Sample count per section (Usage):
Training       28709
PublicTest      3589
PrivateTest     3589

Pixel string length for the first sample: 2304  (= 48*48 check)
```

Class distribution on train (summarized from the chart): Happy is highest (~7215),
Disgust is lowest (~436). This roughly 16x imbalance is one of this dataset's main
challenges.

DataLoader test:
```
train: 28709 samples | first image shape: torch.Size([1, 48, 48]) | first label: 0 (Angry)
val:    3589 samples | first image shape: torch.Size([1, 48, 48]) | first label: 0 (Angry)
test:   3589 samples | first image shape: torch.Size([1, 48, 48]) | first label: 0 (Angry)

One batch from train_loader: images=torch.Size([64, 1, 48, 48]), labels=torch.Size([64])
```

## New Concepts (with a simple example)

- **Dataset and DataLoader in PyTorch**: `Dataset` defines "how to read one sample"
  (like a list you access by index). `DataLoader` wraps a layer around it that groups
  samples into batches, optionally shuffles them, and can load data in parallel with
  multiple workers so the GPU doesn't sit idle waiting.
- **Tensor shape (N, C, H, W)**: in PyTorch, images are stored as (number of samples,
  number of channels, height, width). Here, since images are grayscale, C=1 (one
  channel, unlike RGB where C=3). Our batch output `[64, 1, 48, 48]` means 64 images,
  each with 1 channel and 48x48 pixels.
- **Class imbalance**: when the number of samples per class is very different, the
  model naturally leans toward the majority classes (here, Happy) and performs poorly
  on minority classes (here, Disgust), even if overall accuracy looks good.

## Likely Interview Questions

**1. Why were images stored as pixel strings instead of separate files?**
This format comes from the original FER2013 Kaggle competition; storing everything in
a single CSV makes the dataset one portable file and avoids managing thousands of
separate image files.

**2. Why did you divide pixel values by 255?**
Raw pixel values range from 0 to 255. Normalizing to [0,1] keeps input scale small and
stable, which makes neural network training faster and more stable (more reasonable
gradients).

**3. What problem does class imbalance cause here, and how do you plan to fix it?**
The model could ignore the Disgust class entirely (since it's only ~1.5% of the data)
and still get relatively high accuracy, because mistakes on that class barely matter
to the overall score. We address this in Phase 3 with class weighting (giving more
weight to minority-class loss) and in Phase 4 by reporting per-class F1 (not just
overall accuracy).

**4. Why did you use PublicTest for val and PrivateTest for the final test, not the reverse?**
This naming follows the official Kaggle competition split. PublicTest was visible
during the competition and used for tuning models; PrivateTest was used only at the
end for the final ranking. Following this same convention makes my results comparable
to others' reported results on this dataset.

**5. In the DataLoader, why did you only shuffle train?**
Shuffling train prevents the model from memorizing the data order and overfitting to
it; each epoch sees a different order. For val/test, since we're only evaluating (not
learning), order doesn't matter and shuffling would just be unnecessary overhead.

## Status
Phase 1 complete. Ready for Phase 2 (baseline CNN model).
