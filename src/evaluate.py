"""
ارزیابی نهایی بهترین مدل (checkpoints/best.pt) روی PrivateTest (test set نهایی).
این فایل فقط یک‌بار در پایان پروژه اجرا می‌شود — PrivateTest تا این لحظه هرگز دیده
نشده (نه برای آموزش، نه برای انتخاب checkpoint)، پس عدد این‌جا صادقانه‌ترین برآورد
از عملکرد واقعی مدل روی داده‌ی کاملاً جدید است.

خروجی: confusion matrix، گزارش precision/recall/f1 هر کلاس، و چند نمونه‌ی اشتباه.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import EMOTION_NAMES, get_dataloaders
from model import EmotionCNNImproved

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"استفاده از دستگاه: {device}")

CHECKPOINT_PATH = "checkpoints/best.pt"

model = EmotionCNNImproved().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()
print(f"مدل از {CHECKPOINT_PATH} بارگذاری شد.")

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
print(f"\n=== نتیجه‌ی نهایی روی PrivateTest ===")
print(f"تعداد نمونه: {len(all_labels)}")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

class_names = [EMOTION_NAMES[i] for i in range(len(EMOTION_NAMES))]

# گزارش precision/recall/f1 هر کلاس
report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
print("\n=== گزارش per-class (precision/recall/f1) ===")
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
print("\nConfusion matrix ذخیره شد در results/confusion_matrix.png")

# تحلیل خطا: چند نمونه که مدل با اطمینان بالا اشتباه پیش‌بینی کرده
wrong_mask = all_preds != all_labels
wrong_indices = np.where(wrong_mask)[0]
# مرتب‌سازی بر اساس بالاترین اطمینان اشتباه (بدترین اشتباهات مدل)
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

fig.suptitle("بدترین خطاهای مدل (اشتباه با بالاترین اطمینان)")
fig.tight_layout()
fig.savefig("results/error_analysis.png", dpi=120)
print("نمودار تحلیل خطا ذخیره شد در results/error_analysis.png")
