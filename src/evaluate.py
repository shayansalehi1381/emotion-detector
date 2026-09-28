"""
Final evaluation of the best model (checkpoints/best.pt) on PrivateTest (the final test
set). This file runs only once at the end of the project — PrivateTest has never been
seen until this point (not for training, not for checkpoint selection), so the number
here is the most honest estimate of the model's real performance on completely new data.

Output: confusion matrix, per-class precision/recall/f1 report, and a few misclassified samples.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import EMOTION_NAMES, get_dataloaders
from model import EmotionCNNImproved

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CHECKPOINT_PATH = "checkpoints/best.pt"

model = EmotionCNNImproved().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()
print(f"Model loaded from {CHECKPOINT_PATH}.")

_, _, test_loader = get_dataloaders(batch_size=128, augment=False)

all_preds = []
all_labels = []
all_images = []
all_confidences = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = probs.max(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_confidences.extend(confs.cpu().numpy())
        all_images.extend(images.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_confidences = np.array(all_confidences)

test_accuracy = (all_preds == all_labels).mean()
print(f"\n=== Final result on PrivateTest ===")
print(f"Number of samples: {len(all_labels)}")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

class_names = [EMOTION_NAMES[i] for i in range(len(EMOTION_NAMES))]

# Per-class precision/recall/f1 report
report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
print("\n=== Per-class report (precision/recall/f1) ===")
print(report)

with open("results/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Test accuracy (PrivateTest): {test_accuracy*100:.2f}%\n\n")
    f.write(report)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix - PrivateTest (accuracy={test_accuracy*100:.2f}%)")
fig.tight_layout()
fig.savefig("results/confusion_matrix.png", dpi=120)
print("\nConfusion matrix saved to results/confusion_matrix.png")

# Error analysis: a few samples the model mispredicted with high confidence
wrong_mask = all_preds != all_labels
wrong_indices = np.where(wrong_mask)[0]
# Sort by highest wrong confidence (the model's worst mistakes)
wrong_indices_sorted = wrong_indices[np.argsort(-all_confidences[wrong_indices])]

N_ERRORS = 16
top_errors = wrong_indices_sorted[:N_ERRORS]

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, idx in zip(axes.flat, top_errors):
    img = all_images[idx].squeeze(0)
    true_label = EMOTION_NAMES[all_labels[idx]]
    pred_label = EMOTION_NAMES[all_preds[idx]]
    conf = all_confidences[idx]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"true={true_label}\npred={pred_label} ({conf*100:.0f}%)", fontsize=9)
    ax.axis("off")

fig.suptitle("Worst model mistakes (highest-confidence errors)")
fig.tight_layout()
fig.savefig("results/error_analysis.png", dpi=120)
print("Error analysis chart saved to results/error_analysis.png")
