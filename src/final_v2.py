"""
آزمایش تشخیصی فاز ۳ (final_v2): چرا final_full (وزن کلاس خام + augmentation + scheduler)
از bn_dropout ساده ضعیف‌تر بود؟ چون train_acc در final_full فقط به ~38% رسید — یعنی
underfitting، نه overfitting. فرضیه: وزن خام کلاس Disgust (~9.4x) خیلی تهاجمی بوده و
همراه با augmentation و کاهش سریع LR (patience=3) مدل را بیش‌ازحد محدود کرده.
این نسخه با وزن ملایم‌تر (جذر وزن، smooth=True)، scheduler کم‌تهاجمی‌تر (patience=5)،
و epoch/patience بیشتر دوباره تست می‌کند.
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
print(f"استفاده از دستگاه: {device}")

train_loader, val_loader, _ = get_dataloaders(batch_size=128, augment=True)

class_weights = get_class_weights(smooth=True).to(device)
print(f"وزن کلاس‌های ملایم‌شده (sqrt): {class_weights}")

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
