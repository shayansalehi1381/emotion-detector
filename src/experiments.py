"""
Phase 3: runs several experiments back to back to measure the effect of each
improvement (BatchNorm+Dropout, augmentation, class weighting+scheduler+early
stopping) relative to the Phase 2 baseline. Output: results/experiments_comparison.csv
and a comparison bar chart.
"""

import csv

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

results = []

# The Phase 2 baseline was already run; we record the same number here for comparison
results.append({"name": "baseline (Phase 2)", "val_acc": 0.5684, "epochs_run": 11})


def run_experiment(name, model, train_loader, val_loader, epochs, criterion=None, use_scheduler=False, patience=None):
    print(f"\n{'='*60}\nExperiment: {name}\n{'='*60}")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    safe_name = name.split(" ")[0].lower().replace("+", "_")
    best_val_acc = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        log_path=f"results/train_log_{safe_name}.csv",
        checkpoint_path=f"checkpoints/{safe_name}.pt",
        early_stopping_patience=patience,
    )
    results.append({"name": name, "val_acc": best_val_acc, "epochs_run": epochs})


# Experiment 2: + BatchNorm + Dropout (no augmentation, no class weight)
train_loader, val_loader, _ = get_dataloaders(batch_size=128, augment=False)
model2 = EmotionCNNImproved()
run_experiment("bn_dropout", model2, train_loader, val_loader, epochs=30)

# Experiment 3: + Data Augmentation
train_loader_aug, val_loader, _ = get_dataloaders(batch_size=128, augment=True)
model3 = EmotionCNNImproved()
run_experiment("bn_dropout_aug", model3, train_loader_aug, val_loader, epochs=40)

# Experiment 4 (final): + class weighting + LR scheduler + early stopping
class_weights = get_class_weights().to(device)
criterion_weighted = nn.CrossEntropyLoss(weight=class_weights)
model4 = EmotionCNNImproved()
run_experiment(
    "final_full",
    model4,
    train_loader_aug,
    val_loader,
    epochs=60,
    criterion=criterion_weighted,
    use_scheduler=True,
    patience=10,
)

# Save the comparison table
with open("results/experiments_comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["experiment", "best_val_accuracy", "epochs_run"])
    for r in results:
        writer.writerow([r["name"], f"{r['val_acc']*100:.2f}%", r["epochs_run"]])

print("\n\n=== Comparison table summary ===")
for r in results:
    print(f"{r['name']:25s} | val_acc={r['val_acc']*100:.2f}% | epochs={r['epochs_run']}")
print("\nTable saved to results/experiments_comparison.csv")
