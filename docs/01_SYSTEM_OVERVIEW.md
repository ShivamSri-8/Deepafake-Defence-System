# Ethical Deepfake Defence System
## System Overview & Architecture

---

## 1. Executive Summary

The **Ethical Deepfake Defence System (EDDS)** is a comprehensive AI-powered platform designed for probabilistic detection, forensic analysis, and explainable classification of deepfake media.

### 1.1 Project Scope

| Aspect | Description |
|--------|-------------|
| **Domain** | Computer Vision, Deep Learning, Digital Forensics |
| **Type** | Decision-Support System (Probabilistic) |
| **Target Users** | Researchers, Journalists, Legal Professionals, Security Analysts |
| **Ethical Stance** | Transparency, Explainability, Responsible AI Usage |

### 1.2 Key Differentiators

- **Explainable AI (XAI)**: Every prediction is accompanied by visual and textual explanations
- **Multi-Modal Analysis**: Supports images, videos, and temporal pattern analysis
- **Forensic-Grade Features**: Beyond classification - actual artifact detection
- **Ethical Framework**: Built-in awareness module and responsible usage guidelines
- **Academic Rigor**: No overclaiming, probabilistic outputs with confidence intervals

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │   Web Frontend  │  │  Admin Panel    │  │  API Consumers  │              │
│  │   (React.js)    │  │  (React.js)     │  │  (REST API)     │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
└───────────┼────────────────────┼────────────────────┼────────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API GATEWAY                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Express.js / FastAPI  │  Rate Limiting  │  Auth  │  Request Routing │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MICROSERVICES LAYER                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Media Service  │  │  Detection      │  │  Analytics      │              │
│  │  - Upload       │  │  Service        │  │  Service        │              │
│  │  - Validation   │  │  - Inference    │  │  - Statistics   │              │
│  │  - Preprocessing│  │  - Ensemble     │  │  - Trends       │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
│           │                    │                    │                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Forensics      │  │  Explainability │  │  Ethics         │              │
│  │  Service        │  │  Service        │  │  Service        │              │
│  │  - Landmarks    │  │  - Grad-CAM     │  │  - Awareness    │              │
│  │  - Artifacts    │  │  - LIME         │  │  - Guidelines   │              │
│  │  - Temporal     │  │  - SHAP         │  │  - Reports      │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AI/ML LAYER                                     │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    MODEL INFERENCE ENGINE                          │     │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐ │     │
│  │  │   Xception   │ │ EfficientNet │ │  CNN+LSTM    │ │  Ensemble  │ │     │
│  │  │  (Images)    │ │   (Images)   │ │  (Videos)    │ │  Voting    │ │     │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └────────────┘ │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    FORENSIC ANALYSIS ENGINE                        │     │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐ │     │
│  │  │  Face Mesh   │ │  Blink Rate  │ │  Lip Sync    │ │  Frequency │ │     │
│  │  │  Detection   │ │  Analysis    │ │  Coherence   │ │  Analysis  │ │     │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └────────────┘ │     │
│  └────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │    MongoDB      │  │   Redis Cache   │  │   File Storage  │              │
│  │  - Results      │  │  - Sessions     │  │  - Uploads      │              │
│  │  - Analytics    │  │  - Model Cache  │  │  - Reports      │              │
│  │  - Users        │  │  - Rate Limits  │  │  - Exports      │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Interaction Flow

```
┌──────────┐    ┌────────────┐    ┌─────────────┐    ┌──────────────┐
│  Client  │───▶│   Upload   │───▶│  Preprocess │───▶│   Detect     │
│          │    │   Media    │    │   & Cache   │    │   (Models)   │
└──────────┘    └────────────┘    └─────────────┘    └──────┬───────┘
                                                            │
     ┌──────────────────────────────────────────────────────┘
     ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐
│  Forensic   │───▶│  Explain    │───▶│   Store     │───▶│  Display │
│  Analysis   │    │  (XAI)      │    │   Results   │    │  Report  │
└─────────────┘    └─────────────┘    └─────────────┘    └──────────┘
```

---

## 4. Technology Stack

### Frontend
- **React.js 18+** - Modern UI with hooks and context
- **Chart.js / Recharts** - Data visualization
- **Axios** - HTTP client
- **React Query** - Server state management

### Backend
- **Node.js + Express** - REST API server
- **FastAPI (Python)** - AI inference service
- **JWT** - Authentication

### AI/ML
- **Python 3.9+**
- **PyTorch / TensorFlow**
- **OpenCV** - Image processing
- **MediaPipe** - Face detection/landmarks

### Database
- **MongoDB** - Primary data store
- **Redis** - Caching and session management

---

*Document Version: 1.0 | Created: 2026-02-07*
