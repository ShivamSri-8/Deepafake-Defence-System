# Deepfake Defence System - Handover Instructions

## 🛑 Current Status
The project is **functionally complete** regarding code structure (Frontend, Backend, AI Engine).
- **Frontend**: 100% Complete (React + Vite).
- **Backend**: 100% Complete (Node.js + Express + MongoDB).
- **AI Engine**: Code is 100% Complete, but running in **Hybrid Mode**.
  - ✅ **Xception Model**: Trained and Active (`models/weights/xception_deepfake.h5` exists).
  - ⚠️ **EfficientNet-B4**: Running in Simulation Mode (Missing Weights).
  - ⚠️ **CNN+LSTM**: Running in Simulation Mode (Missing Weights).

---

## 📋 Task List for Next Agent

### 1. Train Missing AI Models (CRITICAL)
 The system currently simulates predictions for EfficientNet and Video Temporal Analysis. To make it fully production-ready, you must train the remaining two models.

**Prerequisite:** Download **FaceForensics++** dataset (images and videos) into `ai-engine/data/`.

#### A. Train EfficientNet-B4 (Image Detection)
*   **Goal:** Create `efficientnet_deepfake.h5`
*   **Command:**
    ```bash
    cd ai-engine
    python training/train_efficientnet.py --data-dir data/images --epochs 30 --batch-size 8
    ```

#### B. Train CNN+LSTM (Video Temporal Analysis)
*   **Goal:** Create `cnn_lstm_deepfake.h5`
*   **Command:**
    ```bash
    cd ai-engine
    python training/train_cnn_lstm.py --data-dir data/videos --epochs 30 --num-frames 20
    ```

---

### 2. Verify Full Ensemble
Once training is complete and files exists in `ai-engine/models/weights/`:
1.  Restart the AI Engine: `python ai-engine/main.py`
2.  Check startup logs for:
    ```
    ✅ Xception model loaded
    ✅ EfficientNet model loaded
    ✅ CNN+LSTM model loaded
    ✅ 3/3 models loaded successfully
    ```

### 3. (Optional) Database Persistence in AI Engine
*   **Current State:** The Node.js backend handles database storage. The Python AI Engine currently computes results but doesn't store them locally in its own DB.
*   **Task:** If independent Python-side persistence is required, implement MongoDB connection in `ai-engine/services/detector.py`. *Note: This is likely unnecessary if using the full web stack.*

---

## 📂 Key Paths
- **Training Scripts:** `ai-engine/training/`
- **Model Weights:** `ai-engine/models/weights/`
- **Dataset Guide:** `ai-engine/DATASET_TRAINING_GUIDE.md`
