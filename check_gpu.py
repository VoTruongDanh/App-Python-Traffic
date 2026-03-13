"""
Quick script to check GPU usage
"""
import torch
from ultralytics import YOLO
import config

print("=" * 50)
print("GPU CHECK")
print("=" * 50)

# Check CUDA
print(f"\n1. CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device Name: {torch.cuda.get_device_name(0)}")
    print(f"   Device Count: {torch.cuda.device_count()}")
    print(f"   Current Device: {torch.cuda.current_device()}")

# Check config
print(f"\n2. Config USE_GPU: {config.USE_GPU}")

# Load a model and check device
print(f"\n3. Loading YOLOv26n model...")
model = YOLO('yolo26n.pt')

# Set device
if config.USE_GPU and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model.to(device)
print(f"   Model device: {next(model.model.parameters()).device}")

# Test inference
print(f"\n4. Testing inference...")
import numpy as np
test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

import time
start = time.time()
results = model(test_frame, verbose=False)
elapsed = time.time() - start

print(f"   Inference time: {elapsed*1000:.1f}ms")
print(f"   Device used: {'GPU (CUDA)' if device == 'cuda' else 'CPU'}")

print("\n" + "=" * 50)
if device == 'cuda':
    print("✅ GPU IS BEING USED!")
else:
    print("⚠️ GPU NOT BEING USED - Running on CPU")
print("=" * 50)
