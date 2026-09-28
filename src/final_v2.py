"""
Phase 3 diagnostic experiment (final_v2): why was final_full (raw class weight +
augmentation + scheduler) worse than plain bn_dropout? Because train_acc in final_full
only reached ~38% — meaning underfitting, not overfitting. Hypothesis: the raw weight
for the Disgust class (~9.4x) was too aggressive and, combined with augmentation and a
fast LR decay (patience=3), over-constrained the model.
This version retests with a milder weight (square-rooted weight, smooth=True), a less
aggressive scheduler (patience=5), and more epochs/patience.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_class_weights, get_dataloaders
from model import EmotionCNNImproved
from train import train_model

SEED = 42
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_loader, val_loader, _ = get_dataloaders(batch_size=128, augment=True)

class_weights = get_class_weights(smooth=True).to(device)
print(f"Smoothed class weights (sqrt): {class_weights}")

model = EmotionCNNImproved()
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=70,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    log_path="results/train_log_final_v2.csv",
    checkpoint_path="checkpoints/final_v2.pt",
    early_stopping_patience=15,
)
