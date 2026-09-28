"""
کلاس Dataset پای‌تورچ برای FER2013. هر ردیف CSV شامل یک رشته‌ی متنی از ۲۳۰۴ عدد
(۴۸×۴۸ پیکسل خاکستری) است که اینجا به آرایه‌ی numpy و سپس تنسور تصویر تبدیل می‌شود.
تقسیم train/val/test از روی ستون Usage انجام می‌شود (تقسیم رسمی خود دیتاست).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

EMOTION_NAMES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

# نگاشت ستون Usage در فایل CSV به نام‌های استاندارد train/val/test
USAGE_MAP = {
    "train": "Training",
    "val": "PublicTest",
    "test": "PrivateTest",
}

IMG_SIZE = 48


class FER2013Dataset(Dataset):
    """
    split: یکی از 'train', 'val', 'test'
    transform: تابع/ترکیب transform های torchvision که روی تصویر PIL یا تنسور اعمال می‌شود
    """

    def __init__(self, csv_path="data/fer2013.csv", split="train", transform=None):
        if split not in USAGE_MAP:
            raise ValueError(f"split باید یکی از {list(USAGE_MAP)} باشد, نه {split!r}")

        df = pd.read_csv(csv_path)
        usage = USAGE_MAP[split]
        self.df = df[df["Usage"] == usage].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # رشته‌ی "70 80 82 ..." را به آرایه‌ی 48x48 uint8 تبدیل می‌کنیم
        pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE)
        label = int(row["emotion"])

        if self.transform is not None:
            image = self.transform(pixels)
        else:
            # پیش‌فرض: تنسور float در بازه‌ی [0, 1] با یک کانال (grayscale)
            image = torch.from_numpy(pixels).float().unsqueeze(0) / 255.0

        return image, label


def get_dataloaders(csv_path="data/fer2013.csv", batch_size=64, num_workers=0, transform=None):
    """سه DataLoader برای train/val/test برمی‌گرداند. فقط train شافل می‌شود."""
    train_ds = FER2013Dataset(csv_path, split="train", transform=transform)
    val_ds = FER2013Dataset(csv_path, split="val", transform=transform)
    test_ds = FER2013Dataset(csv_path, split="test", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # تست سریع صحت: آیا ابعاد و برچسب‌ها منطقی هستند؟
    for split in ["train", "val", "test"]:
        ds = FER2013Dataset(split=split)
        img, label = ds[0]
        print(f"{split}: {len(ds)} نمونه | ابعاد تصویر اول: {img.shape} | برچسب اول: {label} ({EMOTION_NAMES[label]})")

    # تست DataLoader: یک batch واقعی را می‌گیریم و ابعاد آن را چاپ می‌کنیم
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)
    images, labels = next(iter(train_loader))
    print(f"\nیک batch از train_loader: images={images.shape}, labels={labels.shape}")
