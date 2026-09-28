"""
PyTorch Dataset class for FER2013. Each CSV row contains a text string of 2304 numbers
(48x48 grayscale pixels), which here gets converted to a numpy array and then an image
tensor. The train/val/test split is taken from the Usage column (the dataset's own
official split).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

EMOTION_NAMES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

# Maps the CSV file's Usage column to the standard train/val/test names
USAGE_MAP = {
    "train": "Training",
    "val": "PublicTest",
    "test": "PrivateTest",
}

IMG_SIZE = 48


class FER2013Dataset(Dataset):
    """
    split: one of 'train', 'val', 'test'
    transform: a torchvision transform/composition applied to the PIL image or tensor
    """

    def __init__(self, csv_path="data/fer2013.csv", split="train", transform=None):
        if split not in USAGE_MAP:
            raise ValueError(f"split must be one of {list(USAGE_MAP)}, not {split!r}")

        df = pd.read_csv(csv_path)
        usage = USAGE_MAP[split]
        self.df = df[df["Usage"] == usage].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Convert the "70 80 82 ..." string into a 48x48 uint8 array
        pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE)
        label = int(row["emotion"])

        if self.transform is not None:
            image = self.transform(pixels)
        else:
            # default: a float tensor in [0, 1] with one channel (grayscale)
            image = torch.from_numpy(pixels).float().unsqueeze(0) / 255.0

        return image, label


def get_train_transform():
    """
    Data augmentation for train only: horizontal flip, small rotation, and small
    translation. These add artificial variety to the data so the model doesn't overfit
    to exact pixel details. Angles and translations are deliberately kept small since
    the image is only 48x48.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),  # converts to [0,1] and reshapes to (1,H,W)
    ])


def get_eval_transform():
    """We don't want any augmentation for val/test; just a deterministic conversion to a tensor."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])


def get_dataloaders(csv_path="data/fer2013.csv", batch_size=64, num_workers=0, augment=False):
    """
    Returns three DataLoaders for train/val/test. Only train is shuffled.
    augment=True: uses get_train_transform (with augmentation) for train; val/test are
    always evaluated without augmentation (they must be deterministic and reproducible).
    """
    train_transform = get_train_transform() if augment else get_eval_transform()

    train_ds = FER2013Dataset(csv_path, split="train", transform=train_transform)
    val_ds = FER2013Dataset(csv_path, split="val", transform=get_eval_transform())
    test_ds = FER2013Dataset(csv_path, split="test", transform=get_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_class_weights(csv_path="data/fer2013.csv", smooth=False):
    """
    Weight for each class proportional to its inverse frequency in train, to offset
    class imbalance (e.g. Disgust has about 16x fewer samples than Happy).
    Base formula: weight_c = total_count / (num_classes * count of class c)

    smooth=True: uses the square root of the raw weights (weight_c ** 0.5). The raw
    weight for Disgust is about 9.4, which can pull the gradient too aggressively
    toward one small class and cause underfitting; the square root softens that effect
    (down to about 3.07) and gave a more stable result in experiments.
    """
    df = pd.read_csv(csv_path)
    train_df = df[df["Usage"] == "Training"]
    counts = train_df["emotion"].value_counts().sort_index()
    total = len(train_df)
    num_classes = len(counts)
    weights = total / (num_classes * counts)
    if smooth:
        weights = weights ** 0.5
    return torch.tensor(weights.values, dtype=torch.float32)


if __name__ == "__main__":
    # Quick sanity check: are the shapes and labels sensible?
    for split in ["train", "val", "test"]:
        ds = FER2013Dataset(split=split)
        img, label = ds[0]
        print(f"{split}: {len(ds)} samples | first image shape: {img.shape} | first label: {label} ({EMOTION_NAMES[label]})")

    # DataLoader test: grab a real batch and print its shape
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64, augment=True)
    images, labels = next(iter(train_loader))
    print(f"\nOne batch from train_loader: images={images.shape}, labels={labels.shape}")

    weights = get_class_weights()
    print(f"Class weights: {weights}")
