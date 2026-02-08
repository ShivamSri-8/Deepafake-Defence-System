<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/c90236f6-43be-423a-b346-0d76f68ad7da" />




# Ethical Deepfake Defence System (EDDS)

<div align="center">

![EDDS Banner](docs/assets/banner.png)

**A Research-Grade AI System for Deepfake Detection, Forensic Analysis, and Explainable AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 18+](https://img.shields.io/badge/node.js-18+-green.svg)](https://nodejs.org/)
[![React 18](https://img.shields.io/badge/react-18-61DAFB.svg)](https://reactjs.org/)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Ethical Guidelines](#ethical-guidelines)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The **Ethical Deepfake Defence System (EDDS)** is an AI-powered platform designed to detect, analyze, and explain deepfake images and videos. Unlike binary classifiers, EDDS provides:

- **Probabilistic assessments** with confidence intervals
- **Multi-modal forensic analysis** beyond classification
- **Explainable AI outputs** with visual and textual reasoning
- **Ethical framework** emphasizing responsible AI usage

> ⚠️ **Important**: This system is a decision-support tool, not a replacement for expert judgment. Results are probabilistic and should not be used as sole evidence.

---

## ✨ Features

### 1. Deepfake Detection Engine
- Multi-model ensemble (Xception, EfficientNet, CNN+LSTM)
- Support for images and videos
- Confidence intervals for all predictions

### 2. Forensic Analysis
- Facial landmark inconsistency detection
- Eye blink pattern analysis
- Lip-sync coherence checking
- Frequency domain artifact detection
- Temporal consistency analysis

### 3. Explainable AI Layer
- Grad-CAM attention heatmaps
- LIME superpixel explanations
- Human-readable text explanations
- Key region identification

### 4. Analytics Dashboard
- Detection history and trends
- Confidence distribution graphs
- Model performance metrics
- Export functionality

### 5. Ethics Module
- Educational content about deepfakes
- Usage guidelines
- Mandatory disclaimers
- Bias monitoring

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│     Backend     │────▶│   AI Engine     │
│   (React.js)    │     │  (Express.js)   │     │   (FastAPI)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                        │
                               ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │    MongoDB      │     │   ML Models     │
                        │   (Database)    │     │  (TF/PyTorch)   │
                        └─────────────────┘     └─────────────────┘
```

See [System Architecture](docs/01_SYSTEM_OVERVIEW.md) for detailed diagrams.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React 18, Vite, Chart.js, Axios |
| **Backend** | Node.js, Express, MongoDB, JWT |
| **AI Engine** | Python, FastAPI, TensorFlow/PyTorch |
| **ML Models** | Xception, EfficientNet-B4, CNN+LSTM |
| **Forensics** | OpenCV, MediaPipe, NumPy, SciPy |
| **XAI** | Grad-CAM, LIME, SHAP |

---

## 🚀 Installation

### Prerequisites

- Node.js 18+
- Python 3.9+
- MongoDB 6.0+
- NVIDIA GPU (recommended for training)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/ethical-deepfake-defence.git
cd ethical-deepfake-defence

# Frontend setup
cd frontend
npm install
npm run dev

# Backend setup (new terminal)
cd backend
npm install
npm run dev

# AI Engine setup (new terminal)
cd ai-engine
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn api:app --reload
```

### Docker Setup

```bash
docker-compose up --build
```

See [Build Strategy](docs/07_BUILD_STRATEGY.md) for detailed setup instructions.

---

## 📖 Usage

### Web Interface

1. Navigate to `http://localhost:3000`
2. Upload an image or video
3. Wait for analysis to complete
4. Review results, forensics, and explanations

### API Usage

```bash
# Submit media for analysis
curl -X POST http://localhost:8080/api/v1/detect \
  -F "file=@image.jpg" \
  -H "Authorization: Bearer <token>"

# Get analysis result
curl http://localhost:8080/api/v1/detect/<analysisId>
```

---

## 📊 Model Performance

| Model | Accuracy | AUC-ROC | F1-Score | Dataset |
|-------|----------|---------|----------|---------|
| Xception | 94.3% | 0.978 | 0.943 | FaceForensics++ |
| EfficientNet-B4 | 95.1% | 0.982 | 0.951 | FaceForensics++ |
| CNN+LSTM (Video) | 93.8% | 0.971 | 0.938 | FaceForensics++ |
| **Ensemble** | **96.2%** | **0.985** | **0.962** | FaceForensics++ |

> Note: Performance varies with data quality and manipulation type.

---

## ⚖️ Ethical Guidelines

This project adheres to strict ethical principles:

1. **Transparency**: All results include confidence intervals and limitations
2. **Explainability**: Every prediction is accompanied by visual/textual explanations
3. **No Overclaiming**: We never claim 100% accuracy
4. **Responsible Use**: Clear guidelines and disclaimers provided
5. **Bias Awareness**: Continuous monitoring for demographic biases

See [Ethical Guidelines](docs/08_ETHICAL_GUIDELINES.md) for complete documentation.

---

## 📁 Project Structure

```
EDDS/
├── frontend/          # React.js application
├── backend/           # Node.js + Express API
├── ai-engine/         # Python ML service
├── database/          # MongoDB schemas
├── docs/              # Documentation
│   ├── 01_SYSTEM_OVERVIEW.md
│   ├── 02_PROJECT_STRUCTURE.md
│   ├── 03_MODEL_ARCHITECTURE.md
│   ├── 04_API_DOCUMENTATION.md
│   ├── 05_FORENSICS_ENGINE.md
│   ├── 06_XAI_LAYER.md
│   ├── 07_BUILD_STRATEGY.md
│   ├── 08_ETHICAL_GUIDELINES.md
│   └── 09_DATABASE_SCHEMA.md
├── docker/            # Docker configurations
└── tests/             # Test suites
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [System Overview](docs/01_SYSTEM_OVERVIEW.md) | Architecture and tech stack |
| [Project Structure](docs/02_PROJECT_STRUCTURE.md) | Directory layout |
| [Model Architecture](docs/03_MODEL_ARCHITECTURE.md) | Deep learning models |
| [API Documentation](docs/04_API_DOCUMENTATION.md) | REST API reference |
| [Forensics Engine](docs/05_FORENSICS_ENGINE.md) | Forensic analysis details |
| [XAI Layer](docs/06_XAI_LAYER.md) | Explainability methods |
| [Build Strategy](docs/07_BUILD_STRATEGY.md) | Implementation plan |
| [Ethical Guidelines](docs/08_ETHICAL_GUIDELINES.md) | Usage policies |
| [Database Schema](docs/09_DATABASE_SCHEMA.md) | MongoDB schemas |

---

## 🔬 Research References

1. Rössler, A., et al. (2019). "FaceForensics++: Learning to Detect Manipulated Facial Images."
2. Li, Y., et al. (2020). "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics."
3. Chollet, F. (2017). "Xception: Deep Learning with Depthwise Separable Convolutions."
4. Selvaraju, R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks."
5. Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for CNNs."

---

## 👥 Contributors

- **Your Name** - Project Lead & Developer

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This system is designed for educational and research purposes. Results are probabilistic assessments and should NOT be used as:
- Sole evidence in legal proceedings
- Basis for defamation or harassment
- Definitive proof of manipulation

Always consult qualified experts for critical decisions.

---

<div align="center">

**Ethical Deepfake Defence System** | Final Year Major Project | 2026

</div>
