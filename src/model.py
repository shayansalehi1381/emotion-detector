"""
Baseline CNN model for classifying 7 emotions from a 48x48 grayscale image.
This version is deliberately simple (no BatchNorm/Dropout/Augmentation) so Phase 2 has
an honest baseline; improvements are added in Phase 3.
"""

import torch
import torch.nn as nn

NUM_CLASSES = 7


class EmotionCNN(nn.Module):
    """
    Architecture: three (Conv -> ReLU -> MaxPool) blocks that gradually increase
    channel count and shrink the image's spatial dimensions, followed by two
    fully-connected layers for classification.

    Shape path: 48x48x1 -> 24x24x32 -> 12x12x64 -> 6x6x128 -> Flatten(4608) -> 128 -> 7
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


class EmotionCNNImproved(nn.Module):
    """
    Improved version of EmotionCNN for Phase 3: the same three-block architecture,
    plus BatchNorm after every Conv (more stable/faster training) and Dropout before
    the fully-connected layers (reduces overfitting by randomly disabling neurons during train).
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout1=0.5, dropout2=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48 -> 24

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24 -> 12

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 12 -> 6
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout1),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Shape sanity check: does the model's output have the right dimensions?
    model = EmotionCNN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {n_params:,}")

    dummy_input = torch.randn(4, 1, 48, 48)  # 4 test images
    output = model(dummy_input)
    print(f"Input: {dummy_input.shape} -> Output: {output.shape}")
    assert output.shape == (4, NUM_CLASSES), "Model output shape is wrong!"
    print("Output shape test passed.")

    improved = EmotionCNNImproved()
    n_params_improved = sum(p.numel() for p in improved.parameters())
    print(f"\nEmotionCNNImproved parameter count: {n_params_improved:,}")
    out2 = improved(dummy_input)
    assert out2.shape == (4, NUM_CLASSES)
    print(f"Input: {dummy_input.shape} -> Output: {out2.shape}")
