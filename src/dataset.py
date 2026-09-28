"""
کلاس Dataset پای‌تورچ برای FER2013. هر ردیف CSV شامل یک رشته‌ی متنی از ۲۳۰۴ عدد
(۴۸×۴۸ پیکسل خاکستری) است که اینجا به آرایه‌ی numpy و سپس تنسور تصویر تبدیل می‌شود.
تقسیم train/val/test از روی ستون Usage انجام می‌شود (تقسیم رسمی خود دیتاست).
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


def get_train_transform():
    """
    Data augmentation فقط برای train: flip افقی، چرخش کوچک، و جابه‌جایی کوچک.
    این‌ها تنوع مصنوعی به داده اضافه می‌کنند تا مدل به جزئیات دقیق پیکسل‌ها overfit نشود.
    زاویه‌ها و جابه‌جایی‌ها عمداً کوچک نگه داشته شده‌اند چون تصویر فقط 48x48 است.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),  # به [0,1] تبدیل می‌کند و شکل را به (1,H,W) می‌برد
    ])


def get_eval_transform():
    """برای val/test هیچ augmentation نمی‌خواهیم؛ فقط تبدیل قطعی به تنسور."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])


def get_dataloaders(csv_path="data/fer2013.csv", batch_size=64, num_workers=0, augment=False):
    """
    سه DataLoader برای train/val/test برمی‌گرداند. فقط train شافل می‌شود.
    augment=True: روی train از get_train_transform (با augmentation) استفاده می‌کند؛
    val/test همیشه بدون augmentation ارزیابی می‌شوند (باید قطعی و قابل‌تکرار باشند).
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
    وزن هر کلاس متناسب با عکس فراوانی آن در train، برای جبران عدم توازن کلاس‌ها
    (مثلاً Disgust حدود ۱۶ برابر کمتر از Happy نمونه دارد).
    فرمول پایه: weight_c = تعداد_کل / (تعداد_کلاس‌ها * تعداد_نمونه‌ی کلاس c)

    smooth=True: از جذر وزن‌های بالا استفاده می‌کند (weight_c ** 0.5). وزن خام برای
    Disgust حدود 9.4 است که می‌تواند gradient را بیش‌ازحد به سمت یک کلاس کوچک بکشاند
    و باعث underfitting شود؛ جذر آن اثر را ملایم‌تر می‌کند (حدود 3.07) و در آزمایش‌ها
    نتیجه‌ی پایدارتری داد.
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
    # تست سریع صحت: آیا ابعاد و برچسب‌ها منطقی هستند؟
    for split in ["train", "val", "test"]:
        ds = FER2013Dataset(split=split)
        img, label = ds[0]
        print(f"{split}: {len(ds)} نمونه | ابعاد تصویر اول: {img.shape} | برچسب اول: {label} ({EMOTION_NAMES[label]})")

    # تست DataLoader: یک batch واقعی را می‌گیریم و ابعاد آن را چاپ می‌کنیم
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64, augment=True)
    images, labels = next(iter(train_loader))
    print(f"\nیک batch از train_loader: images={images.shape}, labels={labels.shape}")

    weights = get_class_weights()
    print(f"وزن کلاس‌ها: {weights}")
