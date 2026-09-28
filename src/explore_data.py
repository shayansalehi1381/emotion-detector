"""
Initial inspection of the FER2013 dataset: the CSV file's shape, sample count per
section (Usage), and emotion class distribution. Output: a class-distribution chart at
results/class_distribution.png
"""

import pandas as pd
import matplotlib.pyplot as plt

EMOTION_NAMES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

df = pd.read_csv("data/fer2013.csv")

print("DataFrame shape:", df.shape)
print("Columns:", list(df.columns))
print()
print("Sample count per section (Usage):")
print(df["Usage"].value_counts())
print()
print("Pixel string length (should be 48*48=2304 numbers) for the first sample:")
print(len(df.iloc[0]["pixels"].split()))

# Class distribution broken down by train/val/test section
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, usage in zip(axes, ["Training", "PublicTest", "PrivateTest"]):
    subset = df[df["Usage"] == usage]
    counts = subset["emotion"].value_counts().sort_index()
    labels = [EMOTION_NAMES[i] for i in counts.index]
    ax.bar(labels, counts.values, color="steelblue")
    ax.set_title(f"{usage} (n={len(subset)})")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("sample count" if usage == "Training" else "")

fig.suptitle("Emotion class distribution in FER2013")
fig.tight_layout()
fig.savefig("results/class_distribution.png", dpi=120)
print("\nChart saved to results/class_distribution.png")
