"""
رسم نمودار loss و accuracy (train/val) از روی فایل لاگ CSV که train.py تولید می‌کند.
استفاده: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe src/plot_training.py <log_csv> <output_png> [عنوان]
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_log(log_path, output_path, title="روند آموزش"):
    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(df["epoch"], df["train_loss"], label="train loss")
    axes[0].plot(df["epoch"], df["val_loss"], label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_acc"] * 100, label="train accuracy")
    axes[1].plot(df["epoch"], df["val_acc"] * 100, label="val accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    print(f"نمودار ذخیره شد در {output_path}")

    best_val_acc = df["val_acc"].max()
    best_epoch = df.loc[df["val_acc"].idxmax(), "epoch"]
    print(f"بهترین val accuracy: {best_val_acc*100:.2f}% در epoch {int(best_epoch)}")


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "results/train_log_baseline.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "results/baseline_curves.png"
    title = sys.argv[3] if len(sys.argv) > 3 else "روند آموزش baseline"
    plot_training_log(log_path, output_path, title)
