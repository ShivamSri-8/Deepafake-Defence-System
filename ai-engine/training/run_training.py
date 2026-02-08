"""
Complete Training Workflow for EDDS AI Engine
==============================================
This script handles the entire workflow from downloaded dataset to trained models.

Run this after dataset download completes:
    python training/run_training.py

"""
import os
import sys
import zipfile
import shutil
import argparse
from pathlib import Path


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def print_step(step_num, text):
    print(f"\n{'─' * 40}")
    print(f"  Step {step_num}: {text}")
    print(f"{'─' * 40}\n")


def extract_dataset(zip_path, extract_to):
    """Extract the downloaded ZIP file"""
    print(f"📦 Extracting {zip_path}...")
    
    if not os.path.exists(zip_path):
        print(f"❌ ZIP file not found: {zip_path}")
        return False
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"✅ Extracted to: {extract_to}")
    return True


def organize_140k_dataset(source_dir, output_dir):
    """
    Organize the 140k Real and Fake Faces dataset.
    
    Expected structure after download:
    source_dir/
    ├── real_vs_fake/
    │   └── real-vs-fake/
    │       ├── train/
    │       │   ├── real/
    │       │   └── fake/
    │       ├── valid/
    │       │   ├── real/
    │       │   └── fake/
    │       └── test/
    │           ├── real/
    │           └── fake/
    
    We'll combine and organize into:
    output_dir/
    ├── images/
    │   ├── real/
    │   └── fake/
    """
    print(f"📂 Organizing dataset from {source_dir}...")
    
    # Find the actual data directory (handle nested structure)
    data_root = None
    for root, dirs, files in os.walk(source_dir):
        if 'train' in dirs and os.path.exists(os.path.join(root, 'train', 'real')):
            data_root = root
            break
    
    if not data_root:
        # Try alternate structure
        potential_paths = [
            os.path.join(source_dir, 'real_vs_fake', 'real-vs-fake'),
            os.path.join(source_dir, 'real-vs-fake'),
            source_dir
        ]
        for path in potential_paths:
            if os.path.exists(os.path.join(path, 'train')):
                data_root = path
                break
    
    if not data_root:
        print(f"❌ Could not find dataset structure in {source_dir}")
        print("   Looking for: train/real and train/fake subdirectories")
        return False
    
    print(f"✅ Found dataset at: {data_root}")
    
    # Create output directories
    output_real = os.path.join(output_dir, 'images', 'real')
    output_fake = os.path.join(output_dir, 'images', 'fake')
    os.makedirs(output_real, exist_ok=True)
    os.makedirs(output_fake, exist_ok=True)
    
    # Copy images from train, valid, and test
    total_real = 0
    total_fake = 0
    
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            continue
        
        # Real images
        real_dir = os.path.join(split_dir, 'real')
        if os.path.exists(real_dir):
            for img in os.listdir(real_dir):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(real_dir, img)
                    dst = os.path.join(output_real, f"{split}_{img}")
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        total_real += 1
        
        # Fake images
        fake_dir = os.path.join(split_dir, 'fake')
        if os.path.exists(fake_dir):
            for img in os.listdir(fake_dir):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(fake_dir, img)
                    dst = os.path.join(output_fake, f"{split}_{img}")
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        total_fake += 1
        
        print(f"   Processed {split}: {total_real} real, {total_fake} fake images so far...")
    
    print(f"\n✅ Dataset organized:")
    print(f"   Real images: {total_real}")
    print(f"   Fake images: {total_fake}")
    print(f"   Location: {output_dir}/images/")
    
    return True


def check_gpu():
    """Check if GPU is available"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"   - {gpu.name}")
            return True
        else:
            print("⚠️  No GPU detected. Training will use CPU (slower)")
            return False
    except Exception as e:
        print(f"⚠️  Could not check GPU: {e}")
        return False


def train_model(script_path, data_dir, output_path, epochs, batch_size):
    """Train a single model"""
    import subprocess
    
    cmd = [
        sys.executable, script_path,
        "--data-dir", data_dir,
        "--output-path", output_path,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size)
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(script_path)))
    
    if result.returncode == 0:
        print(f"✅ Model saved to: {output_path}")
        return True
    else:
        print(f"❌ Training failed with exit code: {result.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Complete training workflow for EDDS AI Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full workflow (extract, organize, train all models):
  python training/run_training.py

  # Skip extraction if already done:
  python training/run_training.py --skip-extract

  # Only train specific model:
  python training/run_training.py --only xception

  # Quick test with fewer epochs:
  python training/run_training.py --epochs 5 --batch-size 8
        """
    )
    
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing downloaded dataset (default: data)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction (if already extracted)")
    parser.add_argument("--skip-organize", action="store_true",
                        help="Skip organization (if already organized)")
    parser.add_argument("--only", type=str, choices=["xception", "efficientnet", "all"],
                        default="all", help="Which model(s) to train")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16, reduce if OOM)")
    
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / args.data_dir
    zip_file = data_dir / "140k-real-and-fake-faces.zip"
    extracted_dir = data_dir / "140k_extracted"
    organized_dir = data_dir
    training_data = organized_dir / "images"
    weights_dir = base_dir / "models" / "weights"
    
    print_header("EDDS AI Engine Training Workflow")
    
    print("📋 Configuration:")
    print(f"   Data directory: {data_dir}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Models to train: {args.only}")
    
    # Step 0: Check GPU
    print_step(0, "Checking Hardware")
    check_gpu()
    
    # Step 1: Extract ZIP
    if not args.skip_extract:
        print_step(1, "Extracting Dataset")
        if zip_file.exists():
            if not extract_dataset(str(zip_file), str(extracted_dir)):
                print("❌ Extraction failed. Exiting.")
                return 1
        else:
            print(f"⚠️  ZIP file not found: {zip_file}")
            print("   Checking if already extracted...")
            if not extracted_dir.exists():
                print(f"❌ No dataset found. Please download first.")
                return 1
    else:
        print_step(1, "Skipping Extraction (--skip-extract)")
    
    # Step 2: Organize dataset
    if not args.skip_organize:
        print_step(2, "Organizing Dataset")
        
        # Check if already organized
        if training_data.exists() and (training_data / "real").exists():
            real_count = len(list((training_data / "real").glob("*")))
            fake_count = len(list((training_data / "fake").glob("*")))
            if real_count > 0 and fake_count > 0:
                print(f"✅ Dataset already organized:")
                print(f"   Real: {real_count}, Fake: {fake_count}")
            else:
                organize_140k_dataset(str(extracted_dir), str(organized_dir))
        else:
            organize_140k_dataset(str(extracted_dir), str(organized_dir))
    else:
        print_step(2, "Skipping Organization (--skip-organize)")
    
    # Verify training data exists
    if not training_data.exists():
        print(f"❌ Training data not found at: {training_data}")
        return 1
    
    # Step 3: Train models
    print_step(3, "Training Models")
    
    os.makedirs(weights_dir, exist_ok=True)
    training_scripts_dir = base_dir / "training"
    
    models_to_train = []
    if args.only in ["xception", "all"]:
        models_to_train.append({
            "name": "Xception",
            "script": training_scripts_dir / "train_xception.py",
            "output": weights_dir / "xception_deepfake.h5"
        })
    
    if args.only in ["efficientnet", "all"]:
        models_to_train.append({
            "name": "EfficientNet",
            "script": training_scripts_dir / "train_efficientnet.py",
            "output": weights_dir / "efficientnet_deepfake.h5"
        })
    
    results = []
    for i, model_info in enumerate(models_to_train, 1):
        print(f"\n{'━' * 50}")
        print(f"  Training Model {i}/{len(models_to_train)}: {model_info['name']}")
        print(f"{'━' * 50}\n")
        
        success = train_model(
            str(model_info["script"]),
            str(training_data),
            str(model_info["output"]),
            args.epochs,
            args.batch_size
        )
        results.append((model_info["name"], success))
    
    # Summary
    print_header("Training Complete!")
    
    print("📊 Results:")
    for name, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"   {name}: {status}")
    
    print("\n📁 Model weights location:")
    print(f"   {weights_dir}")
    
    if all(r[1] for r in results):
        print("\n🎉 All models trained successfully!")
        print("\n📌 Next steps:")
        print("   1. Start the AI Engine: python main.py")
        print("   2. Check /health/detailed endpoint to verify models are loaded")
        print("   3. Test detection with actual images/videos")
    else:
        print("\n⚠️  Some models failed to train. Check the errors above.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
