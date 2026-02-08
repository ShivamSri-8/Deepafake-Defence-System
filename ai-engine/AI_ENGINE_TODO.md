# AI Engine Implementation Roadmap & Pending Tasks

**Last Updated:** 2026-02-08

This document outlines the remaining work required to move the AI Engine from **Simulation Mode** to a fully functional **Production Mode**.

---

## ✅ COMPLETED

### Forensics Service (`services/forensics.py`)
- [x] **Real Blink Detection (EAR)**:
  - Implemented `_calculate_ear()` for Eye Aspect Ratio calculation
  - Implemented `_get_eye_landmarks()` for extracting MediaPipe eye landmarks
  - Updated `analyze_blink_patterns()` to use real EAR-based blink counting
  - Added EAR consistency analysis (detects unnaturally consistent eye movements)
  
- [x] **FaceMesh Performance Optimization**:
  - Moved FaceMesh initialization outside the loop in `_analyze_video_landmarks()`
  - Now uses single FaceMesh instance for all frames (~10-20x faster)
  - Properly collects per-region scores and deduplicates anomalies

### Detection Service (`services/detector.py`)
- [x] **LSTM Temporal Analysis**:
  - Implemented `_get_lstm_prediction()` method for processing frame sequences
  - Updated `detect_video()` to integrate LSTM predictions with frame analysis
  - Added weighted ensemble (60% frame analysis, 40% temporal analysis)
  - Added detection of frame/temporal disagreement and high variance warnings

### Training Scripts (`training/`)
- [x] **CNN+LSTM Training Script**:
  - Created `training/train_cnn_lstm.py`
  - Uses EfficientNetB0 as CNN feature extractor
  - Dual-layer LSTM for temporal sequence modeling
  - Custom VideoDataGenerator for efficient video loading
  - Two-phase training (frozen CNN → fine-tuned)

### Explainability Engine (`services/explainer.py`)
- [x] **Dynamic Text Explanations**:
  - Rewrote `generate_text_explanation()` to accept real analysis results
  - Now uses detection results, GradCAM focus regions, LIME features, and forensics data
  - Added `_format_region_list()` for natural language formatting
  - Added `_get_region_insight()` for region-specific explanations
  - Added detection of problematic landmark regions
  - Includes blink pattern and temporal jitter findings for videos

- [x] **Data-Driven Key Regions**:
  - Rewrote `_identify_key_regions()` to extract from real GradCAM/LIME results
  - Added `_segment_to_region()` for mapping LIME segments to facial areas
  - Falls back to defaults only when no real data available

- [x] **Updated `explain()` Method**:
  - Now accepts `detection_result` and `forensics_result` parameters
  - Passes all analysis data to text generator for comprehensive explanations

---

## 🔴 Critical: Model Training & Weights

The engine is currently running in simulation mode because model weights are missing.

- [ ] **Train or Download Model Weights**:
  - Train models on a deepfake dataset (e.g., FaceForensics++, DFDC)
  - Or download pre-trained weights from a trusted source
  - Required files in `models/weights/`:
    - `xception_deepfake.h5`
    - `efficientnet_deepfake.h5`
    - `cnn_lstm_deepfake.h5`

### Training Commands
```bash
# Activate virtual environment
cd ai-engine
venv\Scripts\activate

# Train Xception model
python training/train_xception.py --data-dir path/to/dataset --epochs 30

# Train EfficientNet model
python training/train_efficientnet.py --data-dir path/to/dataset --epochs 30

# Train CNN+LSTM model (for videos)
python training/train_cnn_lstm.py --data-dir path/to/video_dataset --epochs 30 --num-frames 20
```

---

## 🟠 Remaining Optional Tasks

### Database Integration
- [ ] **Result Persistence**:
  - `GET /detect/{analysis_id}` currently returns "not found"
  - **Task**: Integrate with MongoDB/PostgreSQL to store and retrieve results

### Testing
- [ ] **Unit Tests**: Add pytest tests for:
  - EAR calculation accuracy
  - Blink detection logic
  - LSTM frame preprocessing
  - Ensemble voting mechanism
  - Text explanation generation

### API Enhancements
- [ ] **Webhook Support**: Add callbacks for async analysis completion
- [ ] **Rate Limiting**: Implement request throttling for production

---

## 🟢 Nice-to-Have Improvements

- [ ] **GPU Acceleration**: Add CUDA/cuDNN optimization checks
- [ ] **Batch Video Processing**: Parallel frame extraction
- [ ] **Caching**: Cache model predictions for repeated analysis
- [ ] **Audio Analysis**: Add audio deepfake detection (lip-sync, voice cloning)
- [ ] **Real-time Processing**: WebSocket support for live video analysis

---

## Project Structure (Current)
```
ai-engine/
├── main.py                    # FastAPI entry point
├── config.py                  # Configuration settings
├── models/
│   ├── schemas.py             # Pydantic models
│   └── weights/               # Model weights (MISSING - CRITICAL)
├── routers/
│   ├── detection.py           # /detect endpoints
│   ├── forensics.py           # /forensics endpoints
│   ├── xai.py                 # /explain endpoints
│   └── health.py              # /health endpoints
├── services/
│   ├── detector.py            # ✅ Deepfake detection (with LSTM)
│   ├── forensics.py           # ✅ Forensic analysis (with EAR)
│   └── explainer.py           # ✅ XAI explanations (data-driven)
├── training/
│   ├── train_xception.py      # Xception training
│   ├── train_efficientnet.py  # EfficientNet training
│   └── train_cnn_lstm.py      # ✅ CNN+LSTM training
└── utils/
    ├── logger.py              # Logging
    ├── file_handler.py        # File uploads
    └── preprocessing.py       # Image/video preprocessing
```

---

## Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Detection Service | ✅ Complete | LSTM integration done |
| Forensics Service | ✅ Complete | EAR blink detection done |
| Explainer Service | ✅ Complete | Data-driven explanations |
| Training Scripts | ✅ Complete | All 3 models have scripts |
| Model Weights | ❌ Missing | **CRITICAL** - Train or download |
| Database | ⏳ Optional | For result persistence |
| Tests | ⏳ Optional | Unit tests for validation |
