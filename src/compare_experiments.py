"""
Final table and chart comparing all Phase 3 experiments (from the CSV log files in results/).
Output: results/experiments_comparison.csv and results/experiments_comparison.png
"""

import matplotlib.pyplot as plt
import pandas as pd

# log filename -> (readable title, short description)
EXPERIMENTS = {
    "results/train_log_baseline.csv": "baseline",
    "results/train_log_bn_dropout.csv": "+BatchNorm+Dropout",
    "results/train_log_bn_dropout_aug.csv": "+Augmentation",
    "results/train_log_final_full.csv": "+ClassWeight(raw)+Scheduler",
    "results/train_log_final_v2.csv": "+ClassWeight(sqrt)+Scheduler",
}

rows = []
for path, label in EXPERIMENTS.items():
    df = pd.read_csv(path)
    best_row = df.loc[df["val_acc"].idxmax()]
    rows.append({
        "experiment": label,
        "best_val_acc": best_row["val_acc"],
        "best_epoch": int(best_row["epoch"]),
        "total_epochs_run": int(df["epoch"].max()),
    })

result_df = pd.DataFrame(rows)
result_df.to_csv("results/experiments_comparison.csv", index=False)
print(result_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["gray"] + ["steelblue"] * (len(result_df) - 2) + ["seagreen"]
bars = ax.bar(result_df["experiment"], result_df["best_val_acc"] * 100, color=colors)
ax.set_ylabel("best val accuracy (%)")
ax.set_title("Phase 3 experiment comparison (vs. Phase 2 baseline)")
ax.tick_params(axis="x", rotation=20)
for bar, val in zip(bars, result_df["best_val_acc"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val*100:.1f}%", ha="center")

fig.tight_layout()
fig.savefig("results/experiments_comparison.png", dpi=120)
print("\nChart saved to results/experiments_comparison.png")
