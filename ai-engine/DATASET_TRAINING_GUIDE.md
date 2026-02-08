# Deepfake Detection Dataset & Training Guide

This guide explains how to obtain datasets and train the models for the EDDS AI Engine.

---

## 📦 Available Datasets

### 1. FaceForensics++ (Recommended for Starting)

**Size:** ~20GB (compressed)  
**Content:** 1,000 original videos + 4 manipulation methods  
**Best For:** Image-based models (Xception, EfficientNet)

#### How to Access:
1. Visit: https://github.com/ondyari/FaceForensics
2. Fill out the Google Form to request access
3. You'll receive a download script via email (usually within 24-48 hours)

#### Kaggle Alternative (Quick Start):
A pre-processed version is available on Kaggle:
- **Dataset:** "FaceForensics++ Faces (C23)" 
- **URL:** https://www.kaggle.com/datasets/sorokin/faceforensics
- **Size:** ~2GB (20,000+ face images, 150x150px)
- **Advantage:** Ready to use, no request needed

```bash
# Download using Kaggle CLI
pip install kaggle
kaggle datasets download -d sorokin/faceforensics
unzip faceforensics.zip -d data/faceforensics
```

---

### 2. Deepfake Detection Challenge (DFDC)

**Size:** ~470GB (full) / ~5GB (preview)  
**Content:** 100,000+ videos with face swaps  
**Best For:** Video models (CNN+LSTM)

#### How to Access:
1. Visit: https://www.kaggle.com/c/deepfake-detection-challenge
2. Accept the competition rules
3. Download from the "Data" tab

#### Preview Dataset (Smaller):
- **Size:** ~5GB
- **Content:** 5,000 videos
- Recommended for testing your training pipeline

```bash
# Download preview dataset
kaggle competitions download -c deepfake-detection-challenge -f dfdc_preview_set.zip
```

---

### 3. Celeb-DF (Celebrity Deepfakes)

**Size:** ~10GB  
**Content:** 590 real + 5,639 synthesized celebrity videos  
**Best For:** High-quality deepfake detection

#### How to Access:
1. Visit: https://github.com/yuezunli/celeb-deepfakeforensics
2. Fill out the request form
3. Download link provided via email

---

### 4. DeeperForensics-1.0

**Size:** ~500GB  
**Content:** 60,000 videos with various perturbations  
**Best For:** Robust detection under real-world conditions

#### How to Access:
1. Visit: https://github.com/EndlessSora/DeeperForensics-1.0
2. Academic/research request required

---

## 🚀 Quick Start: Using Kaggle Dataset

For the fastest setup, use the Kaggle FaceForensics++ dataset:

### Step 1: Setup Kaggle API

```bash
# Install Kaggle CLI
pip install kaggle

# Create API credentials
# Go to kaggle.com -> Account -> Create New API Token
# Save kaggle.json to ~/.kaggle/ (Linux) or C:\Users\<user>\.kaggle\ (Windows)
```

### Step 2: Download Dataset

```bash
cd ai-engine
mkdir -p data

# Download FaceForensics++ faces
kaggle datasets download -d sorokin/faceforensics -p data/
unzip data/faceforensics.zip -d data/faceforensics
```

### Step 3: Organize for Training

The training scripts expect this structure:
```
data/
├── images/              # For Xception/EfficientNet
│   ├── real/
│   │   ├── img001.jpg
│   │   └── ...
│   └── fake/
│       ├── img001.jpg
│       └── ...
└── videos/              # For CNN+LSTM
    ├── real/
    │   ├── video001.mp4
    │   └── ...
    └── fake/
        ├── video001.mp4
        └── ...
```

### Step 4: Prepare Data Script

Create a script to organize the downloaded data:

```python
# prepare_dataset.py
import os
import shutil
from pathlib import Path

def organize_faceforensics(source_dir: str, output_dir: str):
    """Organize FaceForensics++ dataset for training"""
    
    source = Path(source_dir)
    output = Path(output_dir)
    
    # Create directories
    (output / "images" / "real").mkdir(parents=True, exist_ok=True)
    (output / "images" / "fake").mkdir(parents=True, exist_ok=True)
    
    # Move files based on folder structure
    # Adjust based on actual dataset structure
    for label in ["original", "real"]:
        label_dir = source / label
        if label_dir.exists():
            for img in label_dir.glob("*.jpg"):
                shutil.copy(img, output / "images" / "real" / img.name)
    
    for label in ["manipulated", "fake", "Deepfakes", "Face2Face", "FaceSwap"]:
        label_dir = source / label
        if label_dir.exists():
            for img in label_dir.glob("*.jpg"):
                shutil.copy(img, output / "images" / "fake" / img.name)
    
    print(f"Organized dataset in {output}")
    print(f"Real images: {len(list((output / 'images' / 'real').glob('*')))}")
    print(f"Fake images: {len(list((output / 'images' / 'fake').glob('*')))}")

if __name__ == "__main__":
    organize_faceforensics("data/faceforensics", "data")
```

---

## 🏋️ Training the Models

### Prerequisites:

```bash
cd ai-engine
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Train Xception Model:

```bash
python training/train_xception.py \
    --data-dir data/images \
    --output-path models/weights/xception_deepfake.h5 \
    --epochs 30 \
    --batch-size 16
```

**Expected Output:**
- Training time: ~2-4 hours (GPU) / ~12-24 hours (CPU)
- Final accuracy: ~90-95%

### Train EfficientNet Model:

```bash
python training/train_efficientnet.py \
    --data-dir data/images \
    --output-path models/weights/efficientnet_deepfake.h5 \
    --epochs 30 \
    --batch-size 8
```

**Expected Output:**
- Training time: ~3-5 hours (GPU)
- Final accuracy: ~92-97%

### Train CNN+LSTM Model (Videos):

```bash
python training/train_cnn_lstm.py \
    --data-dir data/videos \
    --output-path models/weights/cnn_lstm_deepfake.h5 \
    --epochs 30 \
    --batch-size 4 \
    --num-frames 20
```

**Expected Output:**
- Training time: ~6-12 hours (GPU)
- Final accuracy: ~85-92%

---

## ⚡ GPU Recommendations

| Model | Min VRAM | Recommended | Batch Size |
|-------|----------|-------------|------------|
| Xception | 4GB | 8GB+ | 16-32 |
| EfficientNet-B4 | 6GB | 12GB+ | 8-16 |
| CNN+LSTM | 8GB | 16GB+ | 4-8 |

### Check GPU Availability:

```python
import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices('GPU'))
```

---

## 🔄 Alternative: Pre-trained Weights

If you prefer not to train from scratch, you can search for pre-trained weights:

1. **Hugging Face Hub:** Search for "deepfake detection"
2. **Papers with Code:** Look for model checkpoints
3. **GitHub Repos:** Many research repos include trained weights

**Note:** Always verify the source and license before using third-party weights.

---

## 📊 Expected Model Performance

After training on FaceForensics++:

| Model | Accuracy | AUC | F1-Score |
|-------|----------|-----|----------|
| Xception | 93-96% | 0.97 | 0.94 |
| EfficientNet-B4 | 94-97% | 0.98 | 0.95 |
| CNN+LSTM | 88-93% | 0.94 | 0.90 |
| **Ensemble** | **95-98%** | **0.99** | **0.96** |

---

## ✅ Verification

After training, verify the models work:

```bash
# Start the AI Engine
python main.py

# Visit: http://localhost:8000/health/detailed
# Should show: "models_loaded": true
```

Test with the API:
```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
     -F "file=@test_image.jpg"
```

---

## 🆘 Troubleshooting

### Out of Memory:
- Reduce batch size
- Use mixed precision: `tf.keras.mixed_precision.set_global_policy('mixed_float16')`

### Slow Training:
- Ensure GPU is being used
- Use data generators instead of loading all data

### Low Accuracy:
- Increase epochs
- Add more data augmentationrate
- Try unfreezing more layers
