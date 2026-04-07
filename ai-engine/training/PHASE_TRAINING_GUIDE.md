# 5-PHASE TRAINING PLAN - QUICK REFERENCE GUIDE

## Overview
This plan breaks the Deepfake Detection model training into 5 manageable phases to reduce memory usage, save time, and allow for incremental improvements.

---

## PHASE 1: DATA VALIDATION & PREPARATION ⏱️ 2-3 minutes

**What it does:**
- Validates all 140,000 training images are accessible
- Checks directory structure integrity
- Creates data inventory JSON file
- Confirms labels are balanced

**Run Command:**
```bash
python training/phase_training_plan.py --phase 1
```

**Output:**
- `data_inventory.json` - Complete dataset manifest

**Why this matters:**
- Catches corrupted images early
- Prevents wasted training time on bad data
- Quick sanity check before proceeding

---

## PHASE 2: MODEL ARCHITECTURE & SETUP ⏱️ 1-2 minutes

**What it does:**
- Loads pre-trained Xception model from ImageNet
- Freezes base model layers (transfer learning)
- Adds custom classification head:
  - GlobalAveragePooling2D
  - Dropout(0.5)
  - Dense(512, relu)
  - Dropout(0.3)
  - Dense(1, sigmoid) for binary classification
- Compiles with Adam optimizer

**Run Command:**
```bash
python training/phase_training_plan.py --phase 2
```

**Why this approach:**
- Transfer learning is 90% faster than training from scratch
- Pre-trained ImageNet weights capture general image features
- Only need to train classification head first

**Model Details:**
- Total Parameters: ~22.9M
- Trainable (Phase 2): ~542K (2.4%)
- Frozen: ~22.3M (97.6%)

---

## PHASE 3: QUICK TRAINING (Sanity Check) ⏱️ 5-10 min (GPU) / 30-40 min (CPU)

**What it does:**
- Trains on 1,000 small test images (128×128)
- Quick 3-epoch validation
- Verifies the entire pipeline works
- Tests data generators, augmentation, callbacks
- Minimal GPU/CPU load

**Run Command:**
```bash
python training/phase_training_plan.py --phase 3
```

**Output:**
- `checkpoints/phase_3_quick_train.h5` - Quick checkpoint

**Why this matters:**
- Catches bugs early (before 4+ hour training)
- Validates hardware setup (GPU/CPU)
- Confirms data pipeline works
- Estimates training speed/time

**Expected Results:**
- Accuracy should improve from ~50% → ~70-75%
- AUC should improve rapidly
- If training fails here, it will fail in later phases

---

## PHASE 4: FULL TRAINING - STAGE 1 (Head Training) ⏱️ 2-4 hours (GPU) / 1-2 days (CPU)

**What it does:**
- Trains on full 100,000 training images
- Classification head learns to distinguish real/fake
- Xception base model remains frozen
- 10+ epochs with early stopping
- Uses validation set (20,000 images) for monitoring

**Run Command:**
```bash
python training/phase_training_plan.py --phase 4
```

**Configuration:**
- Batch Size: 32
- Epochs: 10 (auto-stop if no improvement)
- Learning Rate: 0.001
- Augmentation: Full (rotation, shift, zoom, brightness)

**Output:**
- `checkpoints/phase_4_best_head.h5` - Best head-only model
- Training history and validation metrics

**Callbacks Active:**
- **ModelCheckpoint**: Saves best model based on validation AUC
- **EarlyStopping**: Stops if validation AUC doesn't improve for 3 epochs
- **ReduceLROnPlateau**: Reduces learning rate if stuck

**Expected Accuracy:**
- Training Accuracy: ~92-95%
- Validation Accuracy: ~85-90%
- Validation AUC: ~0.95+

---

## PHASE 5: FINE-TUNING (Full Model) ⏱️ 4-8 hours (GPU) / 2-3 days (CPU)

**What it does:**
- Unfreezes last 50 layers of Xception base model
- Trains the entire model end-to-end
- Uses much lower learning rate (0.0001) to fine-tune
- Final polish and optimization
- Tests on all 3 splits (train/valid/test)

**Run Command:**
```bash
python training/phase_training_plan.py --phase 5
```

**Configuration:**
- Batch Size: 16 (smaller batch for fine-tuning)
- Epochs: 15 (with early stopping)
- Learning Rate: 0.0001 (100x lower than Phase 4)
- Unfroze Layers: Last 50 layers of Xception

**Output:**
- `checkpoints/phase_5_best_finetuned.h5` - Best fine-tuned model
- `xception_deepfake_final.h5` - Final production model
- Test set evaluation metrics

**Callbacks Active:**
- **ModelCheckpoint**: Saves best model
- **EarlyStopping**: Patience of 5 epochs
- **ReduceLROnPlateau**: Fine-tuning adjustment

**Expected Results:**
- Final Accuracy: ~92-96%
- Final AUC: ~0.98+
- Production-ready model

---

## QUICK START GUIDE

### Option A: Run All Phases Sequentially
```bash
cd ai-engine
python training/phase_training_plan.py --phase 0
```

### Option B: Run Individual Phases
```bash
# Phase 1 only
python training/phase_training_plan.py --phase 1

# Phase 2 only
python training/phase_training_plan.py --phase 2

# Phase 3 (quick test)
python training/phase_training_plan.py --phase 3

# Phase 4 (full head training)
python training/phase_training_plan.py --phase 4

# Phase 5 (fine-tuning)
python training/phase_training_plan.py --phase 5
```

### Option C: Custom Data Directory
```bash
python training/phase_training_plan.py --phase 0 --data-dir /path/to/dataset
```

---

## TRAINING TIMELINES

### GPU (NVIDIA with CUDA)
| Phase | Duration | Reqs |
|-------|----------|------|
| 1 | 2-3 min | None |
| 2 | 1-2 min | None |
| 3 | 5-10 min | 4GB+ VRAM |
| 4 | 2-4 hours | 8GB+ VRAM |
| 5 | 4-8 hours | 10GB+ VRAM |
| **TOTAL** | **6-12 hours** | - |

### CPU (Standard Laptop/Desktop)
| Phase | Duration | Note |
|-------|----------|------|
| 1 | 2-3 min | Fast on CPU |
| 2 | 1-2 min | Fast on CPU |
| 3 | 30-40 min | Manageable |
| 4 | 1-2 days | SLOW - recommend GPU |
| 5 | 2-3 days | VERY SLOW - GPU strongly recommended |
| **TOTAL** | **3-5 days** | Consider GPU cloud services |

---

## KEY BENEFITS OF 5-PHASE APPROACH

✅ **Memory Efficient**
- Phases 1-3 use minimal memory
- Phase 4 baseline with frozen base
- Phase 5 unfreezes gradually

✅ **Fast Iteration**
- Phase 3 validates pipeline in 5-10 minutes
- Catch bugs before long Phase 4 training
- Early feedback loops

✅ **Production Quality**
- Transfer learning + fine-tuning = best results
- ~96% accuracy achievable
- Lower risk of overfitting

✅ **Checkpoint Management**
- Multiple saved models at each phase
- Can resume from any checkpoint
- Easy rollback to previous best

✅ **Flexible Scheduling**
- Run phases across different days
- Stop/resume without restarting
- Perfect for limited compute resources

---

## TROUBLESHOOTING

### Phase 3 fails?
- Check PyTorch/TensorFlow installation
- Verify GPU drivers (if using GPU)
- Reduce batch size if memory error

### Phase 4 is too slow?
- This is normal on CPU - consider GPU
- Can reduce batch_size (but slower convergence)
- Check if GPU is actually being used

### Phase 5 takes forever?
- Fine-tuning is inherently slow
- Normal behavior on CPU
- GPU is 50-100x faster

### Out of memory during Phase 4/5?
- Reduce batch_size (try 16 or 8)
- Close other applications
- Consider cloud GPU (Google Colab, AWS, etc.)

---

## CHECKPOINT LOCATIONS

All models saved in `ai-engine/training/checkpoints/`:
- `phase_3_quick_train.h5` - Quick sanity check
- `phase_4_best_head.h5` - Best head-only model
- `phase_4_head_trained.h5` - Final head training checkpoint
- `phase_5_best_finetuned.h5` - Best fine-tuned model
- `xception_deepfake_final.h5` - **PRODUCTION MODEL** ⭐

---

## NEXT STEPS AFTER TRAINING

1. **Evaluate Model**
   ```bash
   python services/detector.py --model xception_deepfake_final.h5
   ```

2. **Deploy to API**
   - Copy model to `models/weights/`
   - Update config.py with new model path
   - Restart FastAPI server

3. **Test on Real Videos**
   - Use forensics engine for detailed analysis
   - Generate explainability reports (XAI)

---

## SUCCESS METRICS

✅ **Phase 1:** Data validation passes (no corrupted images)
✅ **Phase 2:** Model compiles successfully
✅ **Phase 3:** Accuracy improves beyond 50% baseline
✅ **Phase 4:** Validation AUC > 0.95
✅ **Phase 5:** Test AUC > 0.98, Accuracy > 94%

---

**Questions? Check:**
- `ai-engine/README.md` - General AI engine docs
- `docs/03_MODEL_ARCHITECTURE.md` - Technical details
- `training/train_xception.py` - Full training reference

Happy Training! 🚀
