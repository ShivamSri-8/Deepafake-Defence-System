# QUICK START: RUN YOUR FIRST PHASE

## ⚡ 60-Second Quick Start

### For Windows:
```batch
cd "c:\Users\HP\Desktop\Deepfake Defence\ai-engine"
python training/phase_training_plan.py --phase 1
```

### For Mac/Linux:
```bash
cd ~/Desktop/Deepfake\ Defence/ai-engine
python training/phase_training_plan.py --phase 1
```

---

## Step-by-Step Instructions

### Step 1: Open Terminal
- **Windows:** PowerShell or CMD
- **Mac:** Terminal
- **Linux:** Any terminal

### Step 2: Navigate to Project
```bash
cd "c:\Users\HP\Desktop\Deepfake Defence"
```

### Step 3: Activate Virtual Environment
```bash
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### Step 4: Run First Phase (Data Validation)
```bash
python ai-engine/training/phase_training_plan.py --phase 1
```

**Expected Output:**
```
======================================================================
PHASE 1: DATA VALIDATION & PREPARATION
======================================================================

  train/fake: 50,000 images
  train/real: 50,000 images
  valid/fake: 10,000 images
  valid/real: 10,000 images
  test/fake: 10,000 images
  test/real: 10,000 images

✓ Data validation COMPLETE - Inventory saved to data_inventory.json
```

**Time Required:** ~2-3 minutes

---

## Next Phases

### Phase 2: Model Setup (2 minutes)
```bash
python ai-engine/training/phase_training_plan.py --phase 2
```

### Phase 3: Quick Test (5-10 min GPU / 30-40 min CPU)
```bash
python ai-engine/training/phase_training_plan.py --phase 3
```

### Phase 4: Full Head Training (2-4 hours GPU)
```bash
python ai-engine/training/phase_training_plan.py --phase 4
```

### Phase 5: Fine-Tuning (4-8 hours GPU)
```bash
python ai-engine/training/phase_training_plan.py --phase 5
```

### Run All Phases
```bash
python ai-engine/training/phase_training_plan.py --phase 0
```

---

## Alternative: Using Batch/Shell Scripts

### Windows (Easier)
```batch
cd ai-engine\training
train.bat 1    # Phase 1
train.bat 3    # Phase 3 (quick test)
train.bat 0    # All phases
```

### Mac/Linux
```bash
cd ai-engine/training
chmod +x train.sh
./train.sh 1   # Phase 1
./train.sh 3   # Phase 3 (quick test)
./train.sh 0   # All phases
```

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'tensorflow'"
```bash
# Install requirements
pip install -r ai-engine/requirements.txt
```

### Error: "File not found"
```bash
# Make sure you're in the right directory
cd "c:\Users\HP\Desktop\Deepfake Defence\ai-engine"

# Check data structure
ls data/140k_extracted/real_vs_fake/real-vs-fake/
```

### Error: "CUDA not found" (GPU users)
```bash
# Check TensorFlow GPU setup
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If no GPU found, training will use CPU (slower but works)
```

---

## Monitoring Progress

### During Phase 3/4/5, you'll see:
```
Epoch 1/10
125/3125 [=>.] - ETA: 15:32 - loss: 0.4521 - accuracy: 0.7821 - auc: 0.9341
```

**What this means:**
- `125/3125` = 125 batches of 3125 total processed
- `ETA: 15:32` = Time remaining
- `loss: 0.4521` = Training loss (lower is better)
- `accuracy: 0.7821` = Accuracy on training batch (78%)
- `auc: 0.9341` = Area under curve (0.93 is excellent)

---

## Estimated Runtimes

### GPU (NVIDIA with CUDA)
| Phase | Time | Hardware |
|-------|------|----------|
| 1 | 2-3 min | Any |
| 2 | 1-2 min | Any |
| 3 | 5-10 min | GPU |
| 4 | 2-4 hours | High-end GPU |
| 5 | 4-8 hours | High-end GPU |

### CPU
| Phase | Time | Note |
|-------|------|------|
| 1-3 | 45 min | Manageable |
| 4-5 | 3-5 days | Very slow, use GPU |

---

## After Training Complete

### Your Models Are Saved Here:
```
ai-engine/training/checkpoints/
├── phase_3_quick_train.h5
├── phase_4_best_head.h5
└── phase_5_best_finetuned.h5

ai-engine/
└── xception_deepfake_final.h5  ⭐ PRODUCTION
```

### Deploy to Production:
```bash
# Copy final model
cp ai-engine/training/xception_deepfake_final.h5 ai-engine/models/weights/

# Your API will now use the new model
```

---

## Need More Info?

- **Full Guide:** `ai-engine/training/PHASE_TRAINING_GUIDE.md`
- **Architecture Details:** `ai-engine/training/TRAINING_ARCHITECTURE.md`
- **AI Engine Docs:** `ai-engine/README.md`
- **Model Details:** `docs/03_MODEL_ARCHITECTURE.md`

---

## Summary

✅ **Phase 1 (Data):** 2-3 min
✅ **Phase 2 (Setup):** 1-2 min
✅ **Phase 3 (Test):** 5-10 min (GPU)
✅ **Phase 4 (Train):** 2-4 hours (GPU)
✅ **Phase 5 (Tune):** 4-8 hours (GPU)

**Total Time:** ~6-12 hours with GPU

**You're ready to train!** 🚀

Run this now:
```bash
python ai-engine/training/phase_training_plan.py --phase 1
```
