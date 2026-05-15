# -*- coding: utf-8 -*-
"""
PyTorch Training Script — EDDS Deepfake Detection
Trains 3 models: Xception, EfficientNet-B4, ResNet50
using BCEWithLogitsLoss + Adam + sigmoid activation

Features:
  - Mixed Precision (FP16) for low-VRAM GPUs (4GB RTX 3050)
  - Checkpoint save/resume (model + optimizer + scaler + epoch)
  - Per-batch progress logging (no more silent epochs)
  - Gradient accumulation for effective larger batch sizes
  - Memory-efficient defaults

Dataset layout expected:
  dataset/
  ├── train/
  │   ├── real/   (real images)
  │   └── fake/   (deepfake images)
  └── val/
      ├── real/
      └── fake/

Usage:
  # Fresh start
  python training/train_pytorch.py --data-dir data --arch efficientnet --epochs 10

  # Resume from checkpoint (auto-detects latest checkpoint)
  python training/train_pytorch.py --data-dir data --arch efficientnet --epochs 10 --resume

  # Resume from specific checkpoint file
  python training/train_pytorch.py --data-dir data --arch efficientnet --epochs 10 --resume --checkpoint models/weights/efficientnet_checkpoint.pt
"""

import os
import sys
import argparse
import time
import math
from pathlib import Path

# Windows fix: num_workers > 0 causes shared-memory error 1455 (paging file too small)
NUM_WORKERS = 0 if sys.platform == "win32" else 4

# Windows fix: force UTF-8 output so Unicode progress bar chars don't crash
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler, autocast

# ── Try importing pretrainedmodels (for Xception) ─────────────────────────────
try:
    import pretrainedmodels
    XCEPTION_AVAILABLE = True
except ImportError:
    XCEPTION_AVAILABLE = False
    print("[WARN] pretrainedmodels not installed - Xception will be skipped.")
    print("   Install with: pip install pretrainedmodels")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Augmentation transforms
TRAIN_TRANSFORMS = T.Compose([
    T.Resize((330, 330)),
    T.RandomCrop((299, 299)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────
def build_xception():
    if not XCEPTION_AVAILABLE:
        raise RuntimeError("pretrainedmodels not installed. Run: pip install pretrainedmodels")
    model = pretrainedmodels.__dict__["xception"](pretrained="imagenet")
    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features, 1)
    return model


def build_efficientnet():
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def build_resnet50():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


MODEL_BUILDERS = {
    "xception":     build_xception,
    "efficientnet": build_efficientnet,
    "resnet50":     build_resnet50,
}

WEIGHT_PATHS = {
    "xception":     "models/weights/xception_deepfake.pt",
    "efficientnet": "models/weights/efficientnet_deepfake.pt",
    "resnet50":     "models/weights/resnet50_deepfake.pt",
}

CHECKPOINT_PATHS = {
    "xception":     "models/weights/xception_checkpoint.pt",
    "efficientnet": "models/weights/efficientnet_checkpoint.pt",
    "resnet50":     "models/weights/resnet50_checkpoint.pt",
}

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(arch, model, optimizer, scaler, scheduler, epoch, best_val_acc, path=None):
    """Save a full training checkpoint for seamless resume."""
    path = path or CHECKPOINT_PATHS[arch]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "arch": arch,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
    }
    torch.save(checkpoint, path)
    print(f"  [CHECKPOINT] Saved -> {path} (epoch {epoch})")


def load_checkpoint(path, model, optimizer, scaler, scheduler):
    """Load a checkpoint and restore all training state."""
    if not os.path.exists(path):
        print(f"  [WARN] Checkpoint not found: {path}")
        return 0, 0.0  # start_epoch, best_val_acc

    print(f"  [RESUME] Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"]
    best_val_acc = checkpoint.get("best_val_acc", 0.0)

    print(f"  [RESUME] Resuming from epoch {start_epoch + 1} (best_val_acc={best_val_acc:.4f})")
    return start_epoch, best_val_acc


# ─────────────────────────────────────────────────────────────────────────────
# Progress bar helper (no tqdm dependency needed)
# ─────────────────────────────────────────────────────────────────────────────
def progress_bar(current, total, prefix="", suffix="", length=30):
    """Print a simple inline progress bar."""
    filled = int(length * current / total)
    bar = "█" * filled + "░" * (length - filled)
    pct = 100 * current / total
    print(f"\r  {prefix} |{bar}| {pct:5.1f}% [{current}/{total}] {suffix}", end="", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop — with AMP, checkpointing, and progress display
# ─────────────────────────────────────────────────────────────────────────────
def train_model(arch: str, data_dir: str, epochs: int, batch_size: int, lr: float,
                resume: bool = False, checkpoint_path: str = None,
                grad_accum_steps: int = 2, log_every: int = 200):
    print(f"\n{'='*60}")
    print(f"  Training: {arch.upper()}")
    print(f"  Device  : {DEVICE}")
    print(f"  Epochs  : {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"  AMP     : {'Enabled (FP16)' if DEVICE.type == 'cuda' else 'Disabled (CPU)'}")
    print(f"  Grad Accum: {grad_accum_steps} steps (effective batch = {batch_size * grad_accum_steps})")
    print(f"{'='*60}\n")

    use_amp = (DEVICE.type == "cuda")

    # ── Data ────────────────────────────────────────────────────────────────
    train_path = os.path.join(data_dir, "train")
    val_path   = os.path.join(data_dir, "val")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}\n"
            "Expected structure:\n"
            "  dataset/train/real/ & dataset/train/fake/\n"
            "  dataset/val/real/   & dataset/val/fake/"
        )

    train_ds = ImageFolder(train_path, transform=TRAIN_TRANSFORMS)
    val_ds   = ImageFolder(val_path,   transform=VAL_TRANSFORMS)

    # ImageFolder assigns class indices alphabetically: fake=0, real=1
    class_to_idx = train_ds.class_to_idx
    print(f"Classes: {class_to_idx}")
    fake_label = class_to_idx.get("fake", 1)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(NUM_WORKERS > 0))
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(NUM_WORKERS > 0))

    total_train_batches = len(train_dl)
    total_val_batches = len(val_dl)

    print(f"[INFO] Training samples  : {len(train_ds):,}  ({total_train_batches:,} batches)")
    print(f"[INFO] Validation samples: {len(val_ds):,}  ({total_val_batches:,} batches)\n")

    # ── Model ────────────────────────────────────────────────────────────────
    model = MODEL_BUILDERS[arch]().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=use_amp)

    save_path = WEIGHT_PATHS[arch]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Resume from checkpoint ──────────────────────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0

    if resume:
        ckpt_path = checkpoint_path or CHECKPOINT_PATHS[arch]
        start_epoch, best_val_acc = load_checkpoint(ckpt_path, model, optimizer, scaler, scheduler)

    # ── Training loop ───────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, epochs + 1):
        t0 = time.time()

        # ── Train phase ─────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        correct = total = 0
        optimizer.zero_grad()

        for batch_idx, (imgs, labels) in enumerate(train_dl, 1):
            imgs = imgs.to(DEVICE, non_blocking=True)
            targets = (labels == fake_label).float().unsqueeze(1).to(DEVICE, non_blocking=True)

            # Mixed precision forward pass
            with autocast(enabled=use_amp):
                output = model(imgs)
                loss = criterion(output, targets) / grad_accum_steps

            # Scaled backward pass
            scaler.scale(loss).backward()

            # Gradient accumulation: step every N batches
            if batch_idx % grad_accum_steps == 0 or batch_idx == total_train_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps
            with torch.no_grad():
                preds = (torch.sigmoid(output) >= 0.5).long().squeeze(1)
                gt = (labels == fake_label).long()
                correct += (preds.cpu() == gt).sum().item()
                total += labels.size(0)

            # Progress display
            if batch_idx % log_every == 0 or batch_idx == total_train_batches:
                running_acc = correct / total if total > 0 else 0
                running_loss = train_loss / batch_idx
                elapsed = time.time() - t0
                eta = elapsed / batch_idx * (total_train_batches - batch_idx)
                progress_bar(
                    batch_idx, total_train_batches,
                    prefix=f"Epoch {epoch:02d} Train",
                    suffix=f"loss={running_loss:.4f} acc={running_acc:.4f} ETA={eta:.0f}s"
                )

            # Explicit memory cleanup every 500 batches
            if batch_idx % 500 == 0:
                torch.cuda.empty_cache()

        train_acc  = correct / total
        train_loss = train_loss / total_train_batches
        print()  # newline after progress bar

        # ── Validate phase ──────────────────────────────────────────────────
        model.eval()
        val_correct = val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(val_dl, 1):
                imgs    = imgs.to(DEVICE, non_blocking=True)
                targets = (labels == fake_label).float().unsqueeze(1).to(DEVICE, non_blocking=True)

                with autocast(enabled=use_amp):
                    output = model(imgs)
                    loss   = criterion(output, targets)

                val_loss += loss.item()
                probs = torch.sigmoid(output).squeeze(1)
                preds = (probs >= 0.5).long()
                gt    = (labels == fake_label).long()
                val_correct += (preds.cpu() == gt).sum().item()
                val_total   += labels.size(0)

                if batch_idx % log_every == 0 or batch_idx == total_val_batches:
                    progress_bar(
                        batch_idx, total_val_batches,
                        prefix=f"Epoch {epoch:02d} Val  ",
                        suffix=f"acc={val_correct/val_total:.4f}"
                    )

        val_acc  = val_correct / val_total
        val_loss = val_loss / total_val_batches
        elapsed  = time.time() - t0
        print()  # newline after progress bar

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"  [{arch}] Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | {elapsed:.1f}s"
        )

        # ── Save best model weights ────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  [BEST] New best model saved -> {save_path} (val_acc={val_acc:.4f})")

        # ── Save checkpoint every epoch for resume ─────────────────────────
        save_checkpoint(arch, model, optimizer, scaler, scheduler, epoch, best_val_acc)

        scheduler.step()

        # Free up VRAM between epochs
        torch.cuda.empty_cache()

    print(f"\n[{arch}] Training complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"[{arch}] Best weights saved to : {save_path}")
    print(f"[{arch}] Last checkpoint saved : {CHECKPOINT_PATHS[arch]}\n")
    return best_val_acc


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train deepfake detection models (Xception / EfficientNet / ResNet50)"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to dataset root (must contain train/ and val/ subdirs with real/ and fake/)"
    )
    parser.add_argument(
        "--arch", type=str, default="all",
        choices=["xception", "efficientnet", "resnet50", "all"],
        help="Which model to train (default: all)"
    )
    parser.add_argument("--epochs",     type=int,   default=30,     help="Total number of epochs (default: 30)")
    parser.add_argument("--batch-size", type=int,   default=4,      help="Batch size (default: 4, optimized for 4GB VRAM)")
    parser.add_argument("--lr",         type=float, default=1e-4,   help="Learning rate (default: 0.0001)")
    parser.add_argument("--resume",     action="store_true",        help="Resume training from latest checkpoint")
    parser.add_argument("--checkpoint", type=str,   default=None,   help="Path to a specific checkpoint file to resume from")
    parser.add_argument("--grad-accum", type=int,   default=2,      help="Gradient accumulation steps (default: 2, effective batch=8)")
    parser.add_argument("--log-every",  type=int,   default=200,    help="Log progress every N batches (default: 200)")

    args = parser.parse_args()

    print(f"\n[EDDS] PyTorch Deepfake Detector Trainer")
    print(f"   Device : {DEVICE}")
    print(f"   Data   : {args.data_dir}")
    print(f"   Models : {args.arch}")
    print(f"   Resume : {args.resume}")
    print(f"   AMP    : {'Enabled' if DEVICE.type == 'cuda' else 'Disabled'}")

    archs = list(MODEL_BUILDERS.keys()) if args.arch == "all" else [args.arch]

    results = {}
    for arch in archs:
        if arch == "xception" and not XCEPTION_AVAILABLE:
            print(f"\n[WARN] Skipping Xception - pretrainedmodels not installed")
            continue
        results[arch] = train_model(
            arch, args.data_dir, args.epochs, args.batch_size, args.lr,
            resume=args.resume,
            checkpoint_path=args.checkpoint,
            grad_accum_steps=args.grad_accum,
            log_every=args.log_every,
        )

    print("\n" + "="*60)
    print("  TRAINING SUMMARY")
    print("="*60)
    for arch, acc in results.items():
        print(f"  {arch:<15} -> Val Accuracy: {acc:.4f}  Saved: {WEIGHT_PATHS[arch]}")
    print("="*60)
    print("\n[DONE] All models trained. Run the AI engine to use them.\n")


if __name__ == "__main__":
    main()
