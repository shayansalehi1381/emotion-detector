"""
مدل CNN پایه برای طبقه‌بندی ۷ حالت احساسی از تصویر ۴۸×۴۸ خاکستری.
این نسخه عمداً ساده است (بدون BatchNorm/Dropout/Augmentation) تا در فاز ۲
یک baseline صادقانه داشته باشیم؛ بهبودها در فاز ۳ اضافه می‌شوند.
"""

import torch
import torch.nn as nn

NUM_CLASSES = 7


class EmotionCNN(nn.Module):
    """
    معماری: سه بلوک (Conv -> ReLU -> MaxPool) که تدریجاً تعداد کانال را زیاد و
    ابعاد مکانی تصویر را کم می‌کنند، سپس دو لایه‌ی fully-connected برای طبقه‌بندی.

    مسیر ابعاد: 48x48x1 -> 24x24x32 -> 12x12x64 -> 6x6x128 -> Flatten(4608) -> 128 -> 7
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48 -> 24

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24 -> 12

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 12 -> 6
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # تست صحت شکل: آیا خروجی مدل ابعاد درستی دارد؟
    model = EmotionCNN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"تعداد پارامترهای مدل: {n_params:,}")

    dummy_input = torch.randn(4, 1, 48, 48)  # 4 تصویر آزمایشی
    output = model(dummy_input)
    print(f"ورودی: {dummy_input.shape} -> خروجی: {output.shape}")
    assert output.shape == (4, NUM_CLASSES), "ابعاد خروجی مدل اشتباه است!"
    print("تست شکل خروجی موفق بود.")
