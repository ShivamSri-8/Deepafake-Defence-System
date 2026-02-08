# Training Scripts

Training scripts for the deepfake detection models.

## Prerequisites

1. **Dataset**: FaceForensics++ or similar deepfake dataset
2. **GPU**: NVIDIA GPU with CUDA support (recommended)
3. **Memory**: At least 16GB RAM, 8GB VRAM

## Dataset Structure

Organize your dataset as follows:
```
dataset/
├── real/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── fake/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

## Training Commands

### Xception Model
```bash
python train_xception.py \
    --data-dir /path/to/dataset \
    --output-path ../models/weights/xception_deepfake.h5 \
    --epochs 30 \
    --batch-size 16
```

### EfficientNet-B4 Model
```bash
python train_efficientnet.py \
    --data-dir /path/to/dataset \
    --output-path ../models/weights/efficientnet_deepfake.h5 \
    --epochs 30 \
    --batch-size 8
```

### CNN+LSTM Model (Video)
```bash
python train_cnn_lstm.py \
    --data-dir /path/to/video_dataset \
    --output-path ../models/weights/cnn_lstm_deepfake.h5 \
    --epochs 20 \
    --sequence-length 20
```

## Expected Performance

| Model | Accuracy | AUC-ROC | Training Time |
|-------|----------|---------|---------------|
| Xception | ~94% | ~0.98 | ~4 hours |
| EfficientNet-B4 | ~95% | ~0.98 | ~6 hours |
| CNN+LSTM | ~93% | ~0.97 | ~8 hours |

## Tips

1. Start with a small subset to verify the pipeline works
2. Use mixed precision training for faster training on modern GPUs
3. Monitor for overfitting using the validation AUC
4. Consider class balancing if your dataset is imbalanced
