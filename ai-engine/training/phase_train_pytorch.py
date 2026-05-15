# -*- coding: utf-8 -*-
"""
PyTorch 8-Phase Progressive Training — EDDS Deepfake Detection
================================================================
Follows the 8-phase plan from OPTIMIZED_8_PHASE_GUIDE.md but uses PyTorch
instead of TensorFlow, producing .pt weights that the AI engine expects.

Phases:
  1   — Data Validation            (~2-3 min)
  2   — Model Setup & Verify       (~1-2 min)
  3   — Sanity Check (1K images)   (~5-10 min GPU) - LIGHTWEIGHT EXAM MODE
  4   — Train on 25% data          (~30-40 min GPU)
  5   — Train on 50% data          (~1 hr GPU)
  6   — Train on 75% data          (~1.5 hrs GPU)
  7   — Train on 100% data         (~2 hrs GPU)
  8   — Fine-tune (unfreeze all)   (~4-8 hrs GPU, OPTIONAL)
  0   — Run ALL phases sequentially

# NOTE FOR EXAM:
# Model accuracy and performance can significantly improve with larger datasets,
# longer training time, and proper hyperparameter tuning.

Models: efficientnet, resnet50, xception (if pretrainedmodels installed)

Usage:
  python training/phase_train_pytorch.py --phase 1
  python training/phase_train_pytorch.py --phase 3 --arch efficientnet
  python training/phase_train_pytorch.py --phase 4a --arch efficientnet --resume
  python training/phase_train_pytorch.py --phase 0 --arch efficientnet   # all phases
"""

import os
import sys
import json
import argparse
import time
import shutil
from pathlib import Path
from datetime import datetime

# Windows fixes
NUM_WORKERS = 0 if sys.platform == "win32" else 4
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torch.amp import GradScaler, autocast

# Try importing pretrainedmodels for Xception
try:
    import pretrainedmodels
    XCEPTION_AVAILABLE = True
except ImportError:
    XCEPTION_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent.parent  # ai-engine/

WEIGHT_PATHS = {
    "xception":     BASE_DIR / "models" / "weights" / "xception_deepfake.pt",
    "efficientnet": BASE_DIR / "models" / "weights" / "efficientnet_deepfake.pt",
    "resnet50":     BASE_DIR / "models" / "weights" / "resnet50_deepfake.pt",
}

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def checkpoint_path(arch, phase):
    return CHECKPOINT_DIR / f"{arch}_phase_{phase}.pt"

def best_weight_path(arch):
    return WEIGHT_PATHS[arch]


# Transforms
TRAIN_TRANSFORMS = T.Compose([
    T.Resize((330, 330)),
    T.RandomCrop((299, 299)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────
def build_xception(freeze_base=True):
    if not XCEPTION_AVAILABLE:
        raise RuntimeError("pretrainedmodels not installed. Run: pip install pretrainedmodels")
    model = pretrainedmodels.__dict__["xception"](pretrained="imagenet")
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features, 1)
    return model


def build_efficientnet(freeze_base=True):
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def build_resnet50(freeze_base=True):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    if freeze_base:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


MODEL_BUILDERS = {
    "xception":     build_xception,
    "efficientnet": build_efficientnet,
    "resnet50":     build_resnet50,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def progress_bar(current, total, prefix="", suffix="", length=30):
    filled = int(length * current / total)
    bar = "█" * filled + "░" * (length - filled)
    pct = 100 * current / total
    print(f"\r  {prefix} |{bar}| {pct:5.1f}% [{current}/{total}] {suffix}", end="", flush=True)


def save_checkpoint_full(arch, phase, model, optimizer, scaler, scheduler, epoch, best_val_acc):
    path = checkpoint_path(arch, phase)
    torch.save({
        "arch": arch,
        "phase": phase,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_acc": best_val_acc,
    }, path)
    print(f"  [CHECKPOINT] Saved -> {path}")


def load_checkpoint_full(path, model, optimizer=None, scaler=None, scheduler=None):
    if not path.exists():
        return 0, 0.0
    print(f"  [RESUME] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch = ckpt.get("epoch", 0)
    best_acc = ckpt.get("best_val_acc", 0.0)
    print(f"  [RESUME] From epoch {epoch}, best_val_acc={best_acc:.4f}")
    return epoch, best_acc


def get_data_subset(dataset, fraction):
    """Return a Subset of the dataset using the given fraction."""
    n = len(dataset)
    k = max(1, int(n * fraction))
    indices = list(range(k))
    return Subset(dataset, indices)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: Data Validation
# ─────────────────────────────────────────────────────────────────────────────
def phase_1(data_dir):
    print("\n" + "=" * 70)
    print("  PHASE 1: DATA VALIDATION")
    print("=" * 70)

    data_path = Path(data_dir)
    inventory = {}
    all_ok = True

    for split in ["train", "val"]:
        split_path = data_path / split
        inventory[split] = {}
        for label in ["real", "fake"]:
            lp = split_path / label
            if lp.exists():
                count = len([f for f in lp.iterdir() if f.is_file()])
                inventory[split][label] = count
                print(f"  {split}/{label}: {count:,} images")
            else:
                print(f"  ❌ MISSING: {lp}")
                all_ok = False

    inv_path = BASE_DIR / "data_inventory.json"
    with open(inv_path, "w") as f:
        json.dump(inventory, f, indent=2)
    print(f"\n  Inventory saved -> {inv_path}")

    if all_ok:
        total = sum(v for s in inventory.values() for v in s.values())
        print(f"\n  ✅ PHASE 1 COMPLETE — {total:,} images validated")
    else:
        print("\n  ❌ PHASE 1 FAILED — fix missing directories above")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: Model Setup
# ─────────────────────────────────────────────────────────────────────────────
def phase_2(arch):
    print("\n" + "=" * 70)
    print(f"  PHASE 2: MODEL SETUP ({arch.upper()})")
    print("=" * 70)

    model = MODEL_BUILDERS[arch](freeze_base=True).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Architecture    : {arch}")
    print(f"  Device          : {DEVICE}")
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Frozen params   : {total_params - trainable_params:,}")
    print(f"\n  ✅ PHASE 2 COMPLETE")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training engine (shared by phases 3, 4a–4d, 5)
# ─────────────────────────────────────────────────────────────────────────────
def train_phase(arch, model, data_dir, phase_name, epochs, batch_size, lr,
                data_fraction=1.0, resume=False, grad_accum=2, log_every=100,
                prev_phase=None):
    """
    Core training loop for a single phase.
    - data_fraction: 0.0–1.0 of training data to use
    - prev_phase: load weights from this phase checkpoint before starting
    """
    use_amp = (DEVICE.type == "cuda")
    print(f"\n{'=' * 70}")
    print(f"  PHASE {phase_name}: Training {arch.upper()}")
    print(f"  Data fraction : {int(data_fraction * 100)}%")
    print(f"  Epochs        : {epochs}")
    print(f"  Batch size    : {batch_size}  |  LR: {lr}")
    print(f"  Grad accum    : {grad_accum} (effective batch={batch_size * grad_accum})")
    print(f"  AMP           : {'Enabled' if use_amp else 'Disabled'}")
    print(f"{'=' * 70}\n")

    # Data loaders
    train_path = Path(data_dir) / "train"
    val_path = Path(data_dir) / "val"

    full_train_ds = ImageFolder(train_path, transform=TRAIN_TRANSFORMS)
    val_ds = ImageFolder(val_path, transform=VAL_TRANSFORMS)

    class_to_idx = full_train_ds.class_to_idx
    fake_label = class_to_idx.get("fake", 0)
    print(f"  Classes: {class_to_idx}  (fake_label={fake_label})")

    # Subset for progressive training
    if data_fraction < 1.0:
        train_ds = get_data_subset(full_train_ds, data_fraction)
    else:
        train_ds = full_train_ds

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))

    total_train = len(train_dl)
    total_val = len(val_dl)
    print(f"  Train samples: {len(train_ds):,}  ({total_train:,} batches)")
    print(f"  Val   samples: {len(val_ds):,}  ({total_val:,} batches)\n")

    # Optimizer etc
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, weight_decay=1e-4)
    # NOTE: Using a cosine annealing scheduler to adjust learning rate over epochs.
    # This helps in converging faster even with a small number of epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=use_amp)

    # DISCLAIMER FOR EXAM DEMO:
    # Model accuracy and performance can significantly improve with larger datasets, 
    # longer training time, and proper hyperparameter tuning.
    print(f"\n  [EXAM NOTICE] Starting lightweight training for demo purposes.")
    print(f"  [EXAM NOTICE] Model accuracy and performance can significantly improve with larger datasets,")
    print(f"                longer training time, and proper hyperparameter tuning.\n")

    start_epoch = 0
    best_val_acc = 0.0

    # Resume from previous phase checkpoint or current phase checkpoint
    if resume:
        ckpt = checkpoint_path(arch, phase_name)
        if ckpt.exists():
            start_epoch, best_val_acc = load_checkpoint_full(ckpt, model, optimizer, scaler, scheduler)
    elif prev_phase:
        prev_ckpt = checkpoint_path(arch, prev_phase)
        if prev_ckpt.exists():
            print(f"  [INIT] Loading weights from phase {prev_phase}")
            prev_data = torch.load(prev_ckpt, map_location=DEVICE, weights_only=False)
            model.load_state_dict(prev_data["model_state_dict"])
            best_val_acc = prev_data.get("best_val_acc", 0.0)
            print(f"  [INIT] Previous best_val_acc={best_val_acc:.4f}")

    # Training loop
    for epoch in range(start_epoch + 1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        correct = total = 0
        optimizer.zero_grad()

        for batch_idx, (imgs, labels) in enumerate(train_dl, 1):
            imgs = imgs.to(DEVICE, non_blocking=True)
            targets = (labels == fake_label).float().unsqueeze(1).to(DEVICE, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                output = model(imgs)
                loss = criterion(output, targets) / grad_accum

            scaler.scale(loss).backward()

            if batch_idx % grad_accum == 0 or batch_idx == total_train:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum
            with torch.no_grad():
                preds = (torch.sigmoid(output) >= 0.5).long().squeeze(1)
                gt = (labels == fake_label).long()
                correct += (preds.cpu() == gt).sum().item()
                total += labels.size(0)

            if batch_idx % log_every == 0 or batch_idx == total_train:
                acc = correct / total if total else 0
                avg_loss = train_loss / batch_idx
                elapsed = time.time() - t0
                eta = elapsed / batch_idx * (total_train - batch_idx)
                progress_bar(batch_idx, total_train,
                             prefix=f"Epoch {epoch:02d} Train",
                             suffix=f"loss={avg_loss:.4f} acc={acc:.4f} ETA={eta:.0f}s")

            if batch_idx % 500 == 0 and DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        train_acc = correct / total
        train_loss_avg = train_loss / total_train
        print()  # newline after progress bar

        # Validation
        model.eval()
        val_correct = val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(val_dl, 1):
                imgs = imgs.to(DEVICE, non_blocking=True)
                targets = (labels == fake_label).float().unsqueeze(1).to(DEVICE, non_blocking=True)

                with autocast("cuda", enabled=use_amp):
                    output = model(imgs)
                    loss = criterion(output, targets)

                val_loss += loss.item()
                preds = (torch.sigmoid(output) >= 0.5).long().squeeze(1)
                gt = (labels == fake_label).long()
                val_correct += (preds.cpu() == gt).sum().item()
                val_total += labels.size(0)

                if batch_idx % log_every == 0 or batch_idx == total_val:
                    progress_bar(batch_idx, total_val,
                                 prefix=f"Epoch {epoch:02d} Val  ",
                                 suffix=f"acc={val_correct / val_total:.4f}")

        val_acc = val_correct / val_total
        val_loss_avg = val_loss / total_val
        elapsed = time.time() - t0
        print()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  [{phase_name}] Epoch {epoch:02d}/{epochs} | "
              f"Train Loss={train_loss_avg:.4f} Acc={train_acc:.4f} | "
              f"Val Loss={val_loss_avg:.4f} Acc={val_acc:.4f} | "
              f"LR={lr_now:.6f} | {elapsed:.1f}s")

        # Save best model weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(best_weight_path(arch)))
            print(f"  ★ NEW BEST — saved -> {best_weight_path(arch)} (val_acc={val_acc:.4f})")

        # Save phase checkpoint
        save_checkpoint_full(arch, phase_name, model, optimizer, scaler, scheduler, epoch, best_val_acc)
        scheduler.step()

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n  ✅ PHASE {phase_name} COMPLETE — Best val accuracy: {best_val_acc:.4f}")
    print(f"     Weights: {best_weight_path(arch)}")
    print(f"     Checkpoint: {checkpoint_path(arch, phase_name)}\n")
    return best_val_acc


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Fine-tune (unfreeze all layers)
# ─────────────────────────────────────────────────────────────────────────────
def phase_5_finetune(arch, model, data_dir, epochs=10, batch_size=4, lr=1e-5,
                     resume=False, grad_accum=4, log_every=100):
    """Unfreeze all layers and fine-tune with a very low LR."""
    print(f"\n{'=' * 70}")
    print(f"  PHASE 5: FINE-TUNING ({arch.upper()}) — Unfreezing all layers")
    print(f"{'=' * 70}")

    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  All {trainable:,} parameters now trainable")
    print(f"  Using very low LR={lr} to avoid catastrophic forgetting\n")

    return train_phase(
        arch, model, data_dir, phase_name="5",
        epochs=epochs, batch_size=batch_size, lr=lr,
        data_fraction=1.0, resume=resume,
        grad_accum=grad_accum, log_every=log_every,
        prev_phase="4d"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="8-Phase Progressive PyTorch Training for EDDS Deepfake Detection"
    )
    parser.add_argument("--phase", type=str, default="0",
                        choices=["0", "1", "2", "3", "4a", "4b", "4c", "4d", "5"],
                        help="Phase to run (0=all)")
    parser.add_argument("--arch", type=str, default="efficientnet",
                        choices=["xception", "efficientnet", "resnet50"],
                        help="Model architecture (default: efficientnet)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Path to dataset root with train/ and val/ subdirs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4, safe for 4GB VRAM)")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from phase checkpoint")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Log progress every N batches")

    args = parser.parse_args()
    phase = args.phase.lower()

    print(f"\n{'=' * 70}")
    print(f"  EDDS PyTorch 8-Phase Progressive Training")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  Arch   : {args.arch}")
    print(f"  Phase  : {phase}")
    print(f"  Resume : {args.resume}")
    print(f"{'=' * 70}")

    if args.arch == "xception" and not XCEPTION_AVAILABLE:
        print("\n❌ Xception requires pretrainedmodels: pip install pretrainedmodels")
        return

    model = None

    # ── Phase 1: Data Validation ─────────────────────────────────────────
    if phase in ["0", "1"]:
        if not phase_1(args.data_dir):
            print("Stopping — fix data issues first.")
            return

    # ── Phase 2: Model Setup ─────────────────────────────────────────────
    if phase in ["0", "2", "3", "4a", "4b", "4c", "4d", "5"]:
        freeze = phase != "5"
        model = MODEL_BUILDERS[args.arch](freeze_base=freeze).to(DEVICE)
        print(f"\n  Model {args.arch} loaded (freeze_base={freeze})")

    # ── Phase 3: Sanity Check (1K images, 3 epochs) ─────────────────────
    # This is the "Exam Mode" — fast training on a subset to produce a working model.
    # We use 1% of the data (approx 1,000 images) and 3 epochs for speed.
    if phase in ["0", "3"]:
        train_phase(
            args.arch, model, args.data_dir, phase_name="3",
            epochs=3, batch_size=args.batch_size, lr=1e-3,
            data_fraction=0.01,  # ~1,000 images from 100K
            resume=args.resume,
            grad_accum=args.grad_accum, log_every=args.log_every,
        )
        
        print("\n" + "*" * 70)
        print("  EXAM READINESS NOTICE:")
        print("  Model weights have been saved to the 'models/weights' directory.")
        print("  The system is now ready for demonstration.")
        print("  NOTE: Model accuracy and performance can significantly improve with")
        print("  larger datasets, longer training time, and proper hyperparameter tuning.")
        print("*" * 70 + "\n")

    # ── Phase 4A: 25% data, 5 epochs ────────────────────────────────────
    if phase in ["0", "4a"]:
        train_phase(
            args.arch, model, args.data_dir, phase_name="4a",
            epochs=5, batch_size=args.batch_size, lr=1e-3,
            data_fraction=0.25,
            resume=args.resume,
            grad_accum=args.grad_accum, log_every=args.log_every,
            prev_phase="3" if phase == "0" else None,
        )

    # ── Phase 4B: 50% data, 5 epochs ────────────────────────────────────
    if phase in ["0", "4b"]:
        train_phase(
            args.arch, model, args.data_dir, phase_name="4b",
            epochs=5, batch_size=args.batch_size, lr=5e-4,
            data_fraction=0.50,
            resume=args.resume,
            grad_accum=args.grad_accum, log_every=args.log_every,
            prev_phase="4a",
        )

    # ── Phase 4C: 75% data, 5 epochs ────────────────────────────────────
    if phase in ["0", "4c"]:
        train_phase(
            args.arch, model, args.data_dir, phase_name="4c",
            epochs=5, batch_size=args.batch_size, lr=3e-4,
            data_fraction=0.75,
            resume=args.resume,
            grad_accum=args.grad_accum, log_every=args.log_every,
            prev_phase="4b",
        )

    # ── Phase 4D: 100% data, 5 epochs ───────────────────────────────────
    if phase in ["0", "4d"]:
        train_phase(
            args.arch, model, args.data_dir, phase_name="4d",
            epochs=5, batch_size=args.batch_size, lr=1e-4,
            data_fraction=1.0,
            resume=args.resume,
            grad_accum=args.grad_accum, log_every=args.log_every,
            prev_phase="4c",
        )

    # ── Phase 5: Fine-tune (unfreeze all) ────────────────────────────────
    if phase in ["0", "5"]:
        phase_5_finetune(
            args.arch, model, args.data_dir,
            epochs=10, batch_size=args.batch_size, lr=1e-5,
            resume=args.resume,
            grad_accum=args.grad_accum * 2,  # double accum for safety
            log_every=args.log_every,
        )

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  TRAINING COMPLETE — {args.arch.upper()}")
    print(f"  Best weights: {best_weight_path(args.arch)}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
