# Pre-trained Model Weights

This directory contains pre-trained model weights for the deepfake detection models.

## Required Files

| File | Model | Size (approx) |
|------|-------|---------------|
| `xception_deepfake.h5` | Xception | ~80MB |
| `efficientnet_deepfake.h5` | EfficientNet-B4 | ~70MB |
| `cnn_lstm_deepfake.h5` | CNN+LSTM | ~100MB |

## Obtaining Weights

### Option 1: Train Your Own
Use the training scripts in the `training/` directory with your dataset.

### Option 2: Download Pre-trained
(Weights will be available on the releases page once trained)

## Without Weights

If weights are not present, the AI engine runs in **simulation mode**:
- Returns synthetic detection results
- Useful for frontend development and testing
- Not suitable for actual deepfake detection

## Dataset for Training

Recommended datasets:
- FaceForensics++ (https://github.com/ondyari/FaceForensics)
- Celeb-DF (https://github.com/yuezunli/celeb-deepfakeforensics)
- DFDC (Deepfake Detection Challenge)
