"""
تست overfit روی یک batch کوچک: اگر مدل نتواند حتی یک batch کوچک (مثلاً ۳۲ نمونه) را
حفظ کند (یعنی loss به نزدیک صفر و accuracy به نزدیک ۱۰۰٪ نرسد)، یعنی در pipeline
(مدل، loss، optimizer) باگ وجود دارد. این تست قبل از هر آموزش طولانی انجام می‌شود.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import FER2013Dataset
from model import EmotionCNN

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"استفاده از دستگاه: {device}")

# یک batch کوچک ثابت از train می‌سازیم
ds = FER2013Dataset(split="train")
BATCH_SIZE = 32
images = torch.stack([ds[i][0] for i in range(BATCH_SIZE)]).to(device)
labels = torch.tensor([ds[i][1] for i in range(BATCH_SIZE)]).to(device)

model = EmotionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

N_STEPS = 200
for step in range(1, N_STEPS + 1):
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if step % 20 == 0 or step == 1:
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        print(f"step {step:3d} | loss={loss.item():.4f} | accuracy روی همین batch={acc*100:.1f}%")

final_acc = (model(images).argmax(dim=1) == labels).float().mean().item()
print(f"\naccuracy نهایی روی batch: {final_acc*100:.1f}%")
if final_acc > 0.95:
    print("تست overfit موفق بود: pipeline (مدل/loss/optimizer) درست کار می‌کند.")
else:
    print("هشدار: مدل نتوانست batch کوچک را حفظ کند. احتمالاً باگی در pipeline هست.")
