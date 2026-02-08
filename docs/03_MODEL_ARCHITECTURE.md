# Deep Learning Model Architecture

---

## 1. Model Selection Justification

| Model | Use Case | Justification |
|-------|----------|---------------|
| **Xception** | Image Classification | Depthwise separable convolutions reduce parameters. Proven 97.8% AUC on FaceForensics++. Efficient for fine-grained face manipulation detection. |
| **EfficientNet-B4** | Image Classification | Compound scaling for optimal depth/width/resolution. State-of-the-art on multiple deepfake benchmarks. Better feature extraction with fewer parameters. |
| **CNN + LSTM** | Video Analysis | CNN extracts spatial features per frame. LSTM captures temporal inconsistencies. Essential for detecting frame-level artifacts. |

---

## 2. Xception-Based Detector

```
Input: 299x299x3 (RGB Image)
    │
    ▼
┌─────────────────────────────────────────┐
│  Entry Flow (Conv + Depthwise Separable)│
│  - 3 Conv blocks with residual          │
│  - Output: 19x19x728                     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Middle Flow (8x Depthwise Separable)   │
│  - Repeated residual blocks             │
│  - Output: 19x19x728                     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Exit Flow                              │
│  - Final depthwise separable blocks     │
│  - Global Average Pooling               │
│  - Output: 2048                          │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Custom Classifier Head                 │
│  - Dropout (0.5)                        │
│  - Dense (512, ReLU)                    │
│  - Dropout (0.3)                        │
│  - Dense (1, Sigmoid)                   │
└─────────────────────────────────────────┘
    │
    ▼
Output: Probability [0, 1] (Real vs Fake)
```

### Implementation

```python
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, Model

def create_xception_detector(input_shape=(299, 299, 3)):
    """
    Create Xception-based deepfake detector.
    """
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze early layers, fine-tune later layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model
```

---

## 3. EfficientNet-B4 Detector

```
Input: 380x380x3 (RGB Image)
    │
    ▼
┌─────────────────────────────────────────┐
│  EfficientNet-B4 Backbone               │
│  - Compound Scaling (α=1.4, β=1.8)      │
│  - MBConv blocks with SE attention      │
│  - Output: 1792-dim feature vector       │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Custom Head                             │
│  - Global Average Pooling               │
│  - BatchNormalization                    │
│  - Dropout (0.4)                        │
│  - Dense (256, ReLU)                    │
│  - Dense (1, Sigmoid)                   │
└─────────────────────────────────────────┘
    │
    ▼
Output: Probability [0, 1]
```

### Implementation

```python
from tensorflow.keras.applications import EfficientNetB4

def create_efficientnet_detector(input_shape=(380, 380, 3)):
    """
    Create EfficientNet-B4 based deepfake detector.
    """
    base_model = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Fine-tune top layers
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    return model
```

---

## 4. CNN + LSTM for Video Analysis

```
Input: Video (N frames, 224x224x3)
    │
    ▼
┌─────────────────────────────────────────┐
│  Frame-Level Feature Extraction         │
│  (EfficientNet-B0 backbone, frozen)     │
│  Output per frame: 1280-dim vector       │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Temporal Sequence (N x 1280)           │
│  - Bidirectional LSTM (256 units)       │
│  - Attention Mechanism                   │
│  - Output: 512-dim temporal encoding    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Classifier                              │
│  - Dense (256, ReLU)                     │
│  - Dropout (0.5)                         │
│  - Dense (1, Sigmoid)                    │
└─────────────────────────────────────────┘
    │
    ▼
Output: Probability [0, 1] + Per-Frame Scores
```

### Implementation

```python
from tensorflow.keras.applications import EfficientNetB0

def create_video_detector(sequence_length=30, frame_size=(224, 224, 3)):
    """
    Create CNN+LSTM video deepfake detector.
    """
    # Frame feature extractor (frozen)
    cnn_base = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        pooling='avg'
    )
    cnn_base.trainable = False
    
    # Time-distributed CNN
    frame_input = layers.Input(shape=(sequence_length, *frame_size))
    x = layers.TimeDistributed(cnn_base)(frame_input)
    
    # Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    
    # Attention
    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    
    # Classifier
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=frame_input, outputs=output)
    return model
```

---

## 5. Ensemble Strategy

```python
class EnsemblePredictor:
    """
    Combines multiple model predictions with weighted voting.
    """
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or {
            'xception': 0.35,
            'efficientnet': 0.35,
            'cnn_lstm': 0.30
        }
    
    def predict(self, image, video_frames=None):
        predictions = {}
        
        # Image models
        predictions['xception'] = self.models['xception'].predict(image)
        predictions['efficientnet'] = self.models['efficientnet'].predict(image)
        
        # Video model (if applicable)
        if video_frames is not None:
            predictions['cnn_lstm'] = self.models['cnn_lstm'].predict(video_frames)
            active_weights = self.weights
        else:
            active_weights = {'xception': 0.5, 'efficientnet': 0.5}
        
        # Weighted average
        final_score = sum(
            predictions[k] * active_weights[k] 
            for k in predictions
        )
        
        # Uncertainty estimation
        scores = list(predictions.values())
        std_dev = np.std(scores)
        confidence_interval = 1.96 * std_dev  # 95% CI
        
        return {
            'probability': float(final_score),
            'confidence_lower': max(0, final_score - confidence_interval),
            'confidence_upper': min(1, final_score + confidence_interval),
            'model_contributions': predictions,
            'agreement_score': 1 - std_dev
        }
```

---

## 6. Training Configuration

```yaml
# training_config.yaml
training:
  batch_size: 32
  epochs: 50
  early_stopping_patience: 10
  learning_rate: 0.0001
  lr_scheduler: cosine_decay
  
augmentation:
  horizontal_flip: true
  rotation_range: 15
  brightness_range: [0.8, 1.2]
  compression_quality: [70, 100]
  
regularization:
  dropout: 0.5
  l2_weight: 0.0001
  label_smoothing: 0.1
  
class_weights:
  real: 1.0
  fake: 1.0  # Adjust if imbalanced
```

---

## 7. Training Datasets

| Dataset | Size | Description | Usage |
|---------|------|-------------|-------|
| FaceForensics++ | 1.8M frames | Multi-method deepfakes | Primary training |
| Celeb-DF (v2) | 5,639 videos | High-quality deepfakes | Generalization testing |
| DFDC | 128K videos | Facebook challenge dataset | Scale testing |
| DeeperForensics-1.0 | 60K videos | Perturbed deepfakes | Robustness testing |

---

## 8. Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| **Precision** | TP/(TP+FP) | Fake detection reliability |
| **Recall** | TP/(TP+FN) | Fake capture rate |
| **F1-Score** | 2×(P×R)/(P+R) | Balanced measure |
| **AUC-ROC** | Area under ROC curve | Threshold-independent |
| **Brier Score** | Mean squared error | Calibration quality |

---

*Document Version: 1.0 | Created: 2026-02-07*
