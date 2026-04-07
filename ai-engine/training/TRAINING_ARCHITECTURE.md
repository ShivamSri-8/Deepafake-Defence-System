# 5-PHASE TRAINING ARCHITECTURE

## Visual Training Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                   5-PHASE TRAINING PIPELINE                      │
│                   Total Time: 6-12 hrs (GPU)                     │
└─────────────────────────────────────────────────────────────────┘

PHASE 1: DATA VALIDATION (2-3 min)
│
├─ Check 140k images exist
├─ Verify directory structure
├─ Create inventory.json
└─ Output: ✓ data_inventory.json

                            ↓

PHASE 2: MODEL SETUP (1-2 min)
│
├─ Load pre-trained Xception (ImageNet)
├─ Add classification head
├─ Freeze base model
└─ Output: Model architecture ready

                            ↓

PHASE 3: SANITY CHECK (5-10 min GPU / 30-40 min CPU)
│
├─ Training Data: 1,000 small images
├─ Training: 3 epochs
├─ Purpose: Verify pipeline works
├─ Expected Acc: 50% → 70-75%
└─ Output: ✓ phase_3_quick_train.h5

                            ↓

PHASE 4: FULL TRAINING (2-4 hrs GPU / 1-2 days CPU)
│
├─ Training Data: 100,000 images
├─ Trainable Layers: Classification head only (2.4%)
├─ Base Model: Frozen (97.6% params)
├─ Epochs: ~10 (with early stopping)
├─ Batch Size: 32
├─ Learning Rate: 0.001
├─ Expected Val Acc: 85-90%
├─ Expected Val AUC: 0.95+
└─ Output: ✓ phase_4_best_head.h5

                            ↓

PHASE 5: FINE-TUNING (4-8 hrs GPU / 2-3 days CPU)
│
├─ Training Data: 100,000 images (ALL)
├─ Trainable Layers: Last 50 of Xception + head (100%)
├─ Base Model: Unfrozen
├─ Epochs: ~15 (with early stopping)
├─ Batch Size: 16 (smaller for fine-tuning)
├─ Learning Rate: 0.0001 (100x lower)
├─ Expected Test Acc: 92-96%
├─ Expected Test AUC: 0.98+
└─ Output: ✓ xception_deepfake_final.h5 (PRODUCTION)

```

---

## Phase Details & Rationale

### PHASE 1: Data Validation
**Why?**
- Prevents wasted training on corrupted data
- Validates 140k image files integrity
- Confirms balanced labels (50k real/fake each)

**Quick Stats:**
- 100k training images (50k real, 50k fake)
- 20k validation images (10k real, 10k fake)  
- 20k test images (10k real, 10k fake)
- Format: JPG, ~25-35 KB each

---

### PHASE 2: Model Architecture
**Architecture:**
```
Input (299×299×3)
    ↓
[Xception - Pre-trained ImageNet]
    • 71 Convolutional Blocks
    • 22.9M Parameters
    • Captured general image features
    ↓
GlobalAveragePooling2D
    • Pool all feature maps to 2048
    ↓
Dropout(0.5)
    • 50% regularization
    ↓
Dense(512, ReLU)
    • Feature transformation
    ↓
Dropout(0.3)
    • 30% regularization
    ↓
Dense(1, Sigmoid)
    • Binary classification output
    ↓
Output: Real/Fake probability [0-1]
```

**Transfer Learning Benefits:**
- Pre-trained weights from 1.2M ImageNet images
- 90% faster convergence than training from scratch
- Only 2.4% of parameters trainable (faster updates)
- Prevents overfitting on 140k samples

---

### PHASE 3: Sanity Check Training
**Purpose:**
- Validates entire pipeline works
- Catches bugs before long training
- Tests data generators & augmentation
- Estimates GPU/CPU speed

**Configuration:**
- Dataset: 1,000 small images (128×128)
- Epochs: 3
- Batch Size: 32
- Time: ~5-10 minutes (GPU), ~30-40 min (CPU)

**Expected Results:**
```
Epoch 1: Loss ~0.70, Acc ~50% (random guessing level)
         → Model is learning basic features

Epoch 2: Loss ~0.35, Acc ~60-70%
         → Transfer learning starting to work

Epoch 3: Loss ~0.20, Acc ~70-75%
         → Good improvement, pipeline validated

✓ If accuracy improves, Phase 4 will work
✗ If accuracy stays at 50%, debug early before Phase 4
```

---

### PHASE 4: Head Training (Full Dataset)
**What's Frozen:**
```
Xception Base Model (22.3M params) → FROZEN
    Don't waste computation on ImageNet weights
    They already understand images perfectly

Classification Head (542K params) → TRAINABLE
    This learns to distinguish REAL from FAKE
    Fast convergence (pre-trained foundation)
```

**Why Start with Frozen Base:**
1. **Speed:** 100x faster than full fine-tuning
2. **Memory:** Requires less GPU RAM
3. **Stability:** Lower risk of divergence
4. **Efficiency:** Head training is sufficient in many cases

**Data & Augmentation:**
```
Training Images: 100,000
Batch Size: 32
Images per epoch: 3,125 batches
Time per epoch: ~15-20 min (GPU), (~2-3 hours CPU)

Augmentation Strategy:
├─ Rotation: ±20°
├─ Shift: ±20% width/height
├─ Zoom: ±20%
├─ Brightness: ±20%
├─ Horizontal Flip: 50%
└─ Purpose: Simulate real-world variations
```

**Performance Tracking:**
```
Validation Set: 20,000 images
├─ Monitored every epoch
├─ Best model saved when AUC improves
└─ Training stops if no improvement for 3 epochs

Expected Convergence:
Epoch 1:  Val Acc ~65%, Val AUC ~0.80
Epoch 3:  Val Acc ~80%, Val AUC ~0.90
Epoch 7:  Val Acc ~87%, Val AUC ~0.95
```

---

### PHASE 5: Full Model Fine-Tuning
**Why Fine-Tune?**
- Phase 4 frozen base plateau at ~87% accuracy
- Need to adapt pre-trained features to deepfake task
- Last 50 Xception layers capture high-level details
- Fine-tuning = 2-3% accuracy improvement → 92-96%

**Unfreezing Strategy:**
```
Xception Base Model:
├─ Early Layers (0-20): FROZEN
│   └─ Low-level features (edges, colors)
│      Already perfect from ImageNet, don't change
│
├─ Late Layers (21-71): UNFROZEN (50 last layers)
│   └─ High-level features (faces, expressions)
│      Need adaptation for deepfake detection
│
Classification Head: TRAINABLE
└─ Still learning deepfake patterns
```

**Critical: Lower Learning Rate**
```
Phase 4: Learning Rate = 0.001
Phase 5: Learning Rate = 0.0001 (100x LOWER)

Why?
├─ Pre-trained weights are already good
├─ Small updates preserve ImageNet knowledge
├─ Large updates would destroy transfer learning
└─ Careful tuning = best performance
```

**Full Dataset Training:**
```
Training: 100,000 images
Validation: 20,000 images
Test: 20,000 images
Batch Size: 16 (smaller batch for stability)
Epochs: ~15 (with early stopping)

Time: 4-8 hours (GPU) / 2-3 days (CPU)
```

**Expected Final Results:**
```
Training Accuracy: 92-96%
Validation Accuracy: 91-94%
Test Accuracy: 92-96%      ← Final metric

Training AUC: 0.98+
Validation AUC: 0.97+
Test AUC: 0.98+           ← Primary metric

False Positive Rate: 2-4%
False Negative Rate: 2-4%
```

---

## Training Efficiency Breakdown

### Why 5 Phases Save Time & Memory

| Aspect | Traditional | 5-Phase Approach | Savings |
|--------|-------------|------------------|---------|
| **Memory** | 10GB+ | 4-8GB | 40-50% |
| **Iteration Speed** | 1 week | 6-12 hours | 170x |
| **Bug Detection** | Late (Day 5) | Early (10 min) | 1000x |
| **Checkpoint Flexibility** | 1 model | 5 checkpoints | 5x |
| **Failed Training Cost** | Everything lost | Only Phase 5 lost | 80% saved |

### GPU vs CPU Times

**GPU (NVIDIA RTX 3090 or equivalent):**
```
Phase 1: 2-3 min
Phase 2: 1-2 min
Phase 3: 5-10 min
Phase 4: 2-4 hours
Phase 5: 4-8 hours
─────────────────
TOTAL: 6-12 hours
```

**CPU (Intel i7/i9 or Apple Silicon):**
```
Phase 1: 2-3 min
Phase 2: 1-2 min
Phase 3: 30-40 min
Phase 4: 1-2 days (very slow)
Phase 5: 2-3 days (very slow)
─────────────────
TOTAL: 3-5 days

RECOMMENDATION: Use GPU or cloud service
- Google Colab: ~4 hours for full training (FREE)
- AWS EC2 g4dn: ~8 hours for full training (~$30)
- Lambda Labs: ~6 hours for full training (~$15)
```

---

## Checkpoint Management

```
ai-engine/training/checkpoints/
│
├─ phase_3_quick_train.h5
│  └─ Size: ~90MB (22.9M base + 542K head)
│     Use: Quick testing, validation
│     Performance: ~75% accuracy
│
├─ phase_4_best_head.h5
│  └─ Size: ~90MB (same model)
│     Use: Production fallback
│     Performance: ~87-90% accuracy
│
├─ phase_4_head_trained.h5
│  └─ Size: ~90MB
│     Use: Reference checkpoint
│     Performance: Latest from Phase 4
│
├─ phase_5_best_finetuned.h5
│  └─ Size: ~90MB
│     Use: Best fine-tuned version
│     Performance: ~95%+ accuracy
│
└─ xception_deepfake_final.h5 ⭐
   └─ Size: ~90MB
      Use: **PRODUCTION DEPLOYMENT**
      Performance: ~96% accuracy, 0.98+ AUC
```

**Deployment:**
```bash
# Copy final model to production weights
cp training/xception_deepfake_final.h5 models/weights/
```

---

## Success Criteria

```
PHASE 1 ✓
└─ All 140k images accessible
   No corrupted files
   Balanced labels confirmed

PHASE 2 ✓
└─ Model compiles without errors
   Parameter count: 22.9M
   Architecture matches specification

PHASE 3 ✓
└─ Accuracy improves from 50% → 70%+
   Early stopping mechanism works
   Data augmentation functioning

PHASE 4 ✓
└─ Validation Accuracy: 85-90%
   Validation AUC: 0.95+
   No memory leaks
   Loss decreasing smoothly

PHASE 5 ✓
└─ Test Accuracy: 92-96%
   Test AUC: 0.98+
   Final Loss: < 0.20
   Model converged succesfully
```

---

## Quick Reference Commands

```bash
# Start Phase 3 (quick test)
cd ai-engine
train.bat 3                    # Windows
./training/train.sh 3          # Linux/Mac

# Run everything
train.bat 0                    # Windows
./training/train.sh 0          # Linux/Mac

# Check specific phase
python training/phase_training_plan.py --phase 4 --data-dir /custom/path
```

---

## Hardware Recommendations

| GPU | Phase 4 Time | Phase 5 Time | Cost |
|-----|------------|------------|------|
| RTX 4090 | 1-2 hours | 2-4 hours | $1600+ |
| RTX 3090 | 2-4 hours | 4-8 hours | $1200+ |
| RTX 3060 Ti | 4-6 hours | 8-12 hours | $400+ |
| Google Colab (FREE) | 3-5 hours | 6-10 hours | $0 |
| AWS g4dn.xlarge | 4-6 hours | 8-12 hours | ~$30 |

**Best Value: Google Colab (FREE with GPU)**
- 12 hours free GPU per session
- Enough for 1 full training cycle
- Already has TensorFlow/PyTorch installed

---

**Next Step:** Run Phase 1 to validate your data!

```bash
python training/phase_training_plan.py --phase 1
```
