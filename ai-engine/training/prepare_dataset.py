"""
Dataset Preparation Script for EDDS AI Engine
Organizes downloaded datasets into the required training format
"""
import os
import shutil
import argparse
from pathlib import Path
from typing import Tuple
import random


def organize_faceforensics_kaggle(source_dir: str, output_dir: str) -> Tuple[int, int]:
    """
    Organize the Kaggle FaceForensics++ dataset.
    
    Expected source structure:
    source_dir/
        original_sequences/...
        manipulated_sequences/...
    OR
    source_dir/
        real/...
        fake/...
    """
    source = Path(source_dir)
    output = Path(output_dir)
    
    # Create output directories
    real_dir = output / "images" / "real"
    fake_dir = output / "images" / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    real_count = 0
    fake_count = 0
    
    # Possible real folder names
    real_patterns = ["original", "real", "original_sequences", "youtube"]
    fake_patterns = ["manipulated", "fake", "deepfakes", "face2face", 
                     "faceswap", "neuraltextures", "manipulated_sequences"]
    
    # Process real images
    for pattern in real_patterns:
        pattern_dir = source / pattern
        if pattern_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img in pattern_dir.rglob(ext):
                    dest = real_dir / f"real_{real_count:06d}{img.suffix}"
                    shutil.copy(img, dest)
                    real_count += 1
    
    # Process fake images
    for pattern in fake_patterns:
        pattern_dir = source / pattern
        if pattern_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img in pattern_dir.rglob(ext):
                    dest = fake_dir / f"fake_{fake_count:06d}{img.suffix}"
                    shutil.copy(img, dest)
                    fake_count += 1
    
    # Also check root for labeled folders
    for subdir in source.iterdir():
        if subdir.is_dir():
            subdir_lower = subdir.name.lower()
            if any(p in subdir_lower for p in real_patterns):
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    for img in subdir.rglob(ext):
                        dest = real_dir / f"real_{real_count:06d}{img.suffix}"
                        shutil.copy(img, dest)
                        real_count += 1
            elif any(p in subdir_lower for p in fake_patterns):
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    for img in subdir.rglob(ext):
                        dest = fake_dir / f"fake_{fake_count:06d}{img.suffix}"
                        shutil.copy(img, dest)
                        fake_count += 1
    
    return real_count, fake_count


def organize_dfdc_videos(source_dir: str, output_dir: str) -> Tuple[int, int]:
    """
    Organize DFDC dataset for video training.
    
    Expected source structure:
    source_dir/
        dfdc_train_part_*/
            *.mp4
            metadata.json (contains labels)
    """
    import json
    
    source = Path(source_dir)
    output = Path(output_dir)
    
    real_dir = output / "videos" / "real"
    fake_dir = output / "videos" / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    real_count = 0
    fake_count = 0
    
    # Process each part folder
    for part_dir in source.glob("dfdc_train_part_*"):
        metadata_path = part_dir / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            for video_name, info in metadata.items():
                video_path = part_dir / video_name
                if video_path.exists():
                    label = info.get("label", "FAKE")
                    
                    if label == "REAL":
                        dest = real_dir / f"real_{real_count:06d}.mp4"
                        shutil.copy(video_path, dest)
                        real_count += 1
                    else:
                        dest = fake_dir / f"fake_{fake_count:06d}.mp4"
                        shutil.copy(video_path, dest)
                        fake_count += 1
    
    return real_count, fake_count


def create_train_val_split(data_dir: str, val_ratio: float = 0.2):
    """
    Create train/validation split directories.
    """
    data = Path(data_dir)
    
    for media_type in ["images", "videos"]:
        media_dir = data / media_type
        if not media_dir.exists():
            continue
        
        for label in ["real", "fake"]:
            source_dir = media_dir / label
            if not source_dir.exists():
                continue
            
            train_dir = media_dir / "train" / label
            val_dir = media_dir / "val" / label
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)
            
            files = list(source_dir.glob("*"))
            random.shuffle(files)
            
            split_idx = int(len(files) * (1 - val_ratio))
            train_files = files[:split_idx]
            val_files = files[split_idx:]
            
            for f in train_files:
                shutil.move(str(f), str(train_dir / f.name))
            
            for f in val_files:
                shutil.move(str(f), str(val_dir / f.name))
            
            print(f"  {media_type}/{label}: {len(train_files)} train, {len(val_files)} val")


def print_dataset_stats(data_dir: str):
    """Print dataset statistics"""
    data = Path(data_dir)
    
    print("\n📊 Dataset Statistics:")
    print("=" * 50)
    
    for media_type in ["images", "videos"]:
        media_dir = data / media_type
        if media_dir.exists():
            print(f"\n{media_type.upper()}:")
            
            for label in ["real", "fake"]:
                label_dir = media_dir / label
                if label_dir.exists():
                    count = len(list(label_dir.glob("*")))
                    print(f"  {label}: {count:,} files")
            
            # Check train/val split
            train_dir = media_dir / "train"
            val_dir = media_dir / "val"
            if train_dir.exists():
                print(f"  Train split exists: {train_dir}")
            if val_dir.exists():
                print(f"  Val split exists: {val_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for EDDS AI Engine training"
    )
    
    parser.add_argument(
        "--source", type=str, required=True,
        help="Source directory containing the downloaded dataset"
    )
    parser.add_argument(
        "--output", type=str, default="data",
        help="Output directory for organized dataset (default: data)"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["faceforensics", "dfdc", "auto"],
        default="auto",
        help="Dataset type (default: auto-detect)"
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Create train/validation split"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="Validation split ratio (default: 0.2)"
    )
    
    args = parser.parse_args()
    
    source = Path(args.source)
    if not source.exists():
        print(f"❌ Source directory not found: {source}")
        return
    
    print(f"📁 Source: {source}")
    print(f"📂 Output: {args.output}")
    
    # Auto-detect dataset type
    dataset_type = args.dataset
    if dataset_type == "auto":
        if any(source.glob("dfdc_train_part_*")):
            dataset_type = "dfdc"
        else:
            dataset_type = "faceforensics"
        print(f"🔍 Auto-detected dataset: {dataset_type}")
    
    # Organize dataset
    print(f"\n🔄 Organizing {dataset_type} dataset...")
    
    if dataset_type == "faceforensics":
        real_count, fake_count = organize_faceforensics_kaggle(args.source, args.output)
        print(f"✅ Organized {real_count:,} real and {fake_count:,} fake images")
    elif dataset_type == "dfdc":
        real_count, fake_count = organize_dfdc_videos(args.source, args.output)
        print(f"✅ Organized {real_count:,} real and {fake_count:,} fake videos")
    
    # Create train/val split if requested
    if args.split:
        print(f"\n📊 Creating train/validation split ({args.val_ratio:.0%} validation)...")
        create_train_val_split(args.output, args.val_ratio)
    
    # Print statistics
    print_dataset_stats(args.output)
    
    print("\n✅ Dataset preparation complete!")
    print("\nNext steps:")
    print("  1. python training/train_xception.py --data-dir data/images")
    print("  2. python training/train_efficientnet.py --data-dir data/images")
    print("  3. python training/train_cnn_lstm.py --data-dir data/videos")


if __name__ == "__main__":
    main()
