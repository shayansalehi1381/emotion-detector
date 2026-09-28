"""
بررسی اولیه‌ی دیتاست FER2013: شکل فایل CSV، تعداد نمونه‌ها در هر بخش (Usage)،
و توزیع کلاس‌های احساسی. خروجی: نمودار توزیع کلاس‌ها در results/class_distribution.png
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

print("شکل دیتافریم:", df.shape)
print("ستون‌ها:", list(df.columns))
print()
print("تعداد نمونه در هر بخش (Usage):")
print(df["Usage"].value_counts())
print()
print("طول رشته‌ی پیکسل (باید 48*48=2304 عدد باشد) برای نمونه‌ی اول:")
print(len(df.iloc[0]["pixels"].split()))

# توزیع کلاس‌ها به تفکیک بخش train/val/test
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, usage in zip(axes, ["Training", "PublicTest", "PrivateTest"]):
    subset = df[df["Usage"] == usage]
    counts = subset["emotion"].value_counts().sort_index()
    labels = [EMOTION_NAMES[i] for i in counts.index]
    ax.bar(labels, counts.values, color="steelblue")
    ax.set_title(f"{usage} (n={len(subset)})")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("تعداد نمونه" if usage == "Training" else "")

fig.suptitle("توزیع کلاس‌های احساسی در FER2013")
fig.tight_layout()
fig.savefig("results/class_distribution.png", dpi=120)
print("\nنمودار ذخیره شد در results/class_distribution.png")
