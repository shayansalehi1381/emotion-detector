"""
Data sanity check: displays a few samples from each emotion class from the train set
and saves them to results/sample_grid.png. Goal: confirm the image reconstruction from
pixel strings was done correctly and labels match the images.
"""

import matplotlib.pyplot as plt

from dataset import EMOTION_NAMES, FER2013Dataset

N_SAMPLES_PER_CLASS = 5

ds = FER2013Dataset(split="train")

# Find the first N samples for each class
samples_by_class = {c: [] for c in EMOTION_NAMES}
for i in range(len(ds)):
    label = int(ds.df.iloc[i]["emotion"])
    if len(samples_by_class[label]) < N_SAMPLES_PER_CLASS:
        samples_by_class[label].append(i)
    if all(len(v) >= N_SAMPLES_PER_CLASS for v in samples_by_class.values()):
        break

n_classes = len(EMOTION_NAMES)
fig, axes = plt.subplots(n_classes, N_SAMPLES_PER_CLASS, figsize=(N_SAMPLES_PER_CLASS * 1.5, n_classes * 1.5))

for row, (label, indices) in enumerate(samples_by_class.items()):
    for col, idx in enumerate(indices):
        img, _ = ds[idx]
        ax = axes[row, col]
        ax.imshow(img.squeeze(0).numpy(), cmap="gray")
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(EMOTION_NAMES[label], rotation=0, labelpad=40, fontsize=10)
            ax.axis("on")
            ax.set_xticks([])
            ax.set_yticks([])

fig.suptitle("Samples from each emotion class (train set)")
fig.tight_layout()
fig.savefig("results/sample_grid.png", dpi=120)
print("Sample grid saved to results/sample_grid.png")
