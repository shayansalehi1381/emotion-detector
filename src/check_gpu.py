"""
تست ساده‌ی راه‌اندازی: بررسی می‌کند که PyTorch به GPU (CUDA) دسترسی دارد یا نه.
اگر این اسکریپت GPU را پیدا نکند، آموزش مدل روی CPU بسیار کند خواهد بود.
"""

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version (build): {torch.version.cuda}")

    # یک تست واقعی: یک تنسور را روی GPU می‌سازیم و ضرب می‌کنیم
    x = torch.rand(1000, 1000, device="cuda")
    y = torch.rand(1000, 1000, device="cuda")
    z = x @ y
    print(f"GPU matmul test OK, result shape: {z.shape}, device: {z.device}")
else:
    print("هشدار: GPU پیدا نشد. آموزش روی CPU انجام می‌شود (بسیار کندتر).")
