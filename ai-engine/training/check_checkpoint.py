"""Quick script to inspect checkpoint files."""
import torch
import os
import sys

weights_dir = os.path.join(os.path.dirname(__file__), "..", "models", "weights")

for fname in os.listdir(weights_dir):
    fpath = os.path.join(weights_dir, fname)
    if fname.endswith("_checkpoint.pt"):
        try:
            ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
            print(f"\n=== {fname} ===")
            print(f"  Arch        : {ckpt.get('arch', 'N/A')}")
            print(f"  Epoch       : {ckpt.get('epoch', 'N/A')}")
            print(f"  Best Val Acc: {ckpt.get('best_val_acc', 'N/A')}")
        except Exception as e:
            print(f"\n=== {fname} === ERROR: {e}")
    elif fname.endswith(".pt"):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"\n=== {fname} === (weights only, {size_mb:.1f} MB)")

# Check if resnet50 weights exist
resnet_path = os.path.join(weights_dir, "resnet50_deepfake.pt")
print(f"\nResNet50 weights exist: {os.path.exists(resnet_path)}")

# Check GPU
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
