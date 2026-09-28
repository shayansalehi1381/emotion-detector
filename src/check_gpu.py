"""
Simple setup test: checks whether PyTorch can access the GPU (CUDA) or not.
If this script can't find a GPU, training the model on CPU will be very slow.
"""

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version (build): {torch.version.cuda}")

    # A real test: build a tensor on the GPU and multiply it
    x = torch.rand(1000, 1000, device="cuda")
    y = torch.rand(1000, 1000, device="cuda")
    z = x @ y
    print(f"GPU matmul test OK, result shape: {z.shape}, device: {z.device}")
else:
    print("Warning: no GPU found. Training will run on CPU (much slower).")
