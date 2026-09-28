"""
Generic training loop for this project's models. The train_model function is reusable
(it's used again in Phase 3 with a different optimizer/scheduler/class-weight).
Running this file directly trains the Phase 2 baseline (without BatchNorm/Dropout/Augmentation).
"""

import csv
import time

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import EmotionCNN

SEED = 42
torch.manual_seed(SEED)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """Runs one epoch. If train=True, weights are updated."""
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=30,
    lr=1e-3,
    criterion=None,
    optimizer=None,
    scheduler=None,
    log_path="results/train_log.csv",
    checkpoint_path="checkpoints/model.pt",
    early_stopping_patience=None,
):
    """
    Trains the model + CSV log per epoch + saves the best checkpoint (based on val accuracy).
    If early_stopping_patience is given, training stops early if val accuracy doesn't
    improve for that many epochs (used in Phase 3).
    """
    model = model.to(device)
    criterion = criterion or nn.CrossEntropyLoss()
    optimizer = optimizer or optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    epochs_without_improvement = 0

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"])

        for epoch in range(1, epochs + 1):
            start = time.time()

            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step(val_loss) if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) else scheduler.step()

            elapsed = time.time() - start
            print(
                f"epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
                f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% | lr={current_lr:.6f} | {elapsed:.1f}s"
            )
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, current_lr, elapsed])
            f.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_without_improvement += 1

            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
                break

    print(f"\nBest val accuracy: {best_val_acc*100:.2f}% | checkpoint saved to {checkpoint_path}")
    return best_val_acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(batch_size=128)

    model = EmotionCNN()

    train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=30,
        lr=1e-3,
        log_path="results/train_log_baseline.csv",
        checkpoint_path="checkpoints/baseline.pt",
    )
