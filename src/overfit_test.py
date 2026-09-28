"""
Overfit test on a small batch: if the model can't even memorize a small batch (e.g. 32
samples) — i.e. loss doesn't approach zero and accuracy doesn't approach 100% — there's
a bug somewhere in the pipeline (model, loss, optimizer). This test runs before any
long training.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import FER2013Dataset
from model import EmotionCNN

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# build a fixed small batch from train
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
        print(f"step {step:3d} | loss={loss.item():.4f} | accuracy on this batch={acc*100:.1f}%")

final_acc = (model(images).argmax(dim=1) == labels).float().mean().item()
print(f"\nFinal accuracy on the batch: {final_acc*100:.1f}%")
if final_acc > 0.95:
    print("Overfit test passed: the pipeline (model/loss/optimizer) works correctly.")
else:
    print("Warning: the model failed to memorize the small batch. There's likely a bug in the pipeline.")
