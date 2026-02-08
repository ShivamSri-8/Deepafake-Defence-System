# EDDS AI Engine

The AI Engine component of the Ethical Deepfake Defence System. Built with FastAPI, TensorFlow, and PyTorch.

## Features

- **Multi-Model Ensemble Detection**: Combines Xception, EfficientNet-B4, and CNN+LSTM models
- **Forensic Analysis**: Facial landmarks, blink detection, frequency analysis, temporal consistency
- **Explainable AI**: Grad-CAM heatmaps, LIME superpixel explanations, human-readable text
- **REST API**: Full FastAPI documentation with Swagger UI

## Quick Start

### Prerequisites

- Python 3.9+
- NVIDIA GPU (recommended for model inference)

### Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Development mode (with auto-reload)
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
ai-engine/
├── main.py              # FastAPI application entry point
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
├── models/
│   ├── schemas.py       # Pydantic request/response models
│   └── weights/         # Pre-trained model weights (not included)
├── routers/
│   ├── detection.py     # Detection endpoints
│   ├── forensics.py     # Forensic analysis endpoints
│   ├── xai.py           # Explainability endpoints
│   └── health.py        # Health check endpoints
├── services/
│   ├── detector.py      # Deepfake detection service
│   ├── forensics.py     # Forensic analysis service
│   └── explainer.py     # XAI explanation service
├── utils/
│   ├── logger.py        # Logging utilities
│   ├── file_handler.py  # File upload handling
│   └── preprocessing.py # Image/video preprocessing
└── uploads/             # Uploaded files (created automatically)
```

## API Endpoints

### Detection
- `POST /api/v1/detect` - Analyze image/video for deepfakes
- `POST /api/v1/detect/batch` - Batch analysis (up to 10 files)
- `GET /api/v1/detect/{analysis_id}` - Get previous result

### Forensics
- `POST /api/v1/forensics/analyze` - Full forensic analysis
- `POST /api/v1/forensics/landmarks` - Facial landmark analysis
- `POST /api/v1/forensics/frequency` - Frequency domain analysis
- `POST /api/v1/forensics/blink` - Blink pattern analysis (video only)

### Explainability (XAI)
- `POST /api/v1/explain` - Full explanation generation
- `POST /api/v1/explain/gradcam` - Grad-CAM heatmap
- `POST /api/v1/explain/lime` - LIME explanation
- `POST /api/v1/explain/text` - Text explanation

### Health
- `GET /health` - Basic health check
- `GET /health/detailed` - System info and GPU status

## Model Weights

Pre-trained model weights are not included due to size. To enable real detection:

1. Train models using the training scripts (see `training/` directory)
2. Or download pre-trained weights from the release page
3. Place weights in `models/weights/`:
   - `xception_deepfake.h5`
   - `efficientnet_deepfake.h5`
   - `cnn_lstm_deepfake.h5`

Without model weights, the engine runs in **simulation mode** with synthetic results.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `True` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `FAKE_THRESHOLD` | Threshold for fake classification | `0.5` |
| `XCEPTION_WEIGHT` | Xception model weight | `0.35` |
| `EFFICIENTNET_WEIGHT` | EfficientNet weight | `0.40` |
| `LSTM_WEIGHT` | CNN+LSTM weight | `0.25` |

## License

MIT License - See root LICENSE file
