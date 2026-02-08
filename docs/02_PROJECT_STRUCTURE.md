# Module-Wise Project Structure

---

## Complete Directory Structure

```
EDDS/
├── 📁 frontend/                     # React.js Application
│   ├── 📁 public/
│   │   └── index.html
│   ├── 📁 src/
│   │   ├── 📁 components/           # Reusable UI Components
│   │   │   ├── 📁 common/           # Buttons, Cards, Modals, Loaders
│   │   │   ├── 📁 detection/        # Upload, Results Display, Progress
│   │   │   ├── 📁 forensics/        # Heatmaps, Landmark Visualization
│   │   │   ├── 📁 analytics/        # Charts, Graphs, Tables
│   │   │   ├── 📁 ethics/           # Awareness Components
│   │   │   └── 📁 layout/           # Header, Footer, Sidebar, Navigation
│   │   ├── 📁 pages/                # Page Components
│   │   │   ├── HomePage.jsx
│   │   │   ├── DetectionPage.jsx
│   │   │   ├── HistoryPage.jsx
│   │   │   ├── AnalyticsPage.jsx
│   │   │   ├── ForensicsPage.jsx
│   │   │   ├── EthicsPage.jsx
│   │   │   └── AdminPage.jsx
│   │   ├── 📁 hooks/                # Custom React Hooks
│   │   │   ├── useDetection.js
│   │   │   ├── useAnalytics.js
│   │   │   └── useWebSocket.js
│   │   ├── 📁 context/              # Global State Management
│   │   │   ├── AuthContext.jsx
│   │   │   └── ThemeContext.jsx
│   │   ├── 📁 services/             # API Integration
│   │   │   ├── api.js
│   │   │   ├── detectionService.js
│   │   │   └── analyticsService.js
│   │   ├── 📁 utils/                # Helper Functions
│   │   │   ├── formatters.js
│   │   │   └── validators.js
│   │   ├── 📁 assets/               # Static Assets
│   │   │   ├── 📁 images/
│   │   │   └── 📁 icons/
│   │   ├── 📁 styles/               # CSS/SCSS Files
│   │   │   ├── globals.css
│   │   │   └── variables.css
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── 📁 backend/                      # Node.js + Express API
│   ├── 📁 src/
│   │   ├── 📁 controllers/          # Route Handlers
│   │   │   ├── detectionController.js
│   │   │   ├── analyticsController.js
│   │   │   ├── historyController.js
│   │   │   └── ethicsController.js
│   │   ├── 📁 services/             # Business Logic
│   │   │   ├── mediaService.js
│   │   │   ├── aiProxyService.js
│   │   │   └── reportService.js
│   │   ├── 📁 models/               # MongoDB Schemas
│   │   │   ├── Analysis.js
│   │   │   ├── Analytics.js
│   │   │   └── ModelMetrics.js
│   │   ├── 📁 middleware/           # Express Middleware
│   │   │   ├── auth.js
│   │   │   ├── upload.js
│   │   │   ├── validation.js
│   │   │   └── rateLimit.js
│   │   ├── 📁 routes/               # API Routes
│   │   │   ├── detection.js
│   │   │   ├── analytics.js
│   │   │   ├── history.js
│   │   │   └── index.js
│   │   ├── 📁 utils/                # Helpers
│   │   │   ├── logger.js
│   │   │   └── responseHandler.js
│   │   └── app.js
│   ├── 📁 config/
│   │   └── config.js
│   ├── server.js
│   └── package.json
│
├── 📁 ai-engine/                    # Python ML Service
│   ├── 📁 models/                   # Trained Model Files
│   │   ├── 📁 xception/
│   │   │   └── xception_deepfake.h5
│   │   ├── 📁 efficientnet/
│   │   │   └── efficientnet_b4.h5
│   │   └── 📁 cnn_lstm/
│   │       └── video_detector.h5
│   ├── 📁 src/
│   │   ├── 📁 detection/            # Inference Pipeline
│   │   │   ├── base_detector.py
│   │   │   ├── xception_detector.py
│   │   │   ├── efficientnet_detector.py
│   │   │   └── video_detector.py
│   │   ├── 📁 forensics/            # Forensic Analyzers
│   │   │   ├── face_analyzer.py
│   │   │   ├── blink_detector.py
│   │   │   ├── lip_sync_analyzer.py
│   │   │   ├── frequency_analyzer.py
│   │   │   └── temporal_analyzer.py
│   │   ├── 📁 explainability/       # XAI Implementations
│   │   │   ├── gradcam.py
│   │   │   ├── lime_explainer.py
│   │   │   └── text_generator.py
│   │   ├── 📁 preprocessing/        # Data Transforms
│   │   │   ├── image_processor.py
│   │   │   ├── video_processor.py
│   │   │   └── face_extractor.py
│   │   ├── 📁 ensemble/             # Model Fusion
│   │   │   └── ensemble_predictor.py
│   │   └── 📁 utils/
│   │       ├── config.py
│   │       └── helpers.py
│   ├── 📁 training/                 # Model Training Scripts
│   │   ├── train_xception.py
│   │   ├── train_efficientnet.py
│   │   ├── train_cnn_lstm.py
│   │   ├── evaluate_model.py
│   │   └── 📁 configs/
│   │       └── training_config.yaml
│   ├── api.py                       # FastAPI Service
│   ├── requirements.txt
│   └── Dockerfile
│
├── 📁 database/                     # Database Configuration
│   ├── 📁 schemas/                  # MongoDB Schema Definitions
│   │   └── schemas.md
│   ├── 📁 migrations/               # Data Migrations
│   └── 📁 seeds/                    # Sample Data
│       └── seed_data.js
│
├── 📁 docs/                         # Documentation
│   ├── 01_SYSTEM_OVERVIEW.md
│   ├── 02_PROJECT_STRUCTURE.md
│   ├── 03_MODEL_ARCHITECTURE.md
│   ├── 04_API_DOCUMENTATION.md
│   ├── 05_FORENSICS_ENGINE.md
│   ├── 06_XAI_LAYER.md
│   ├── 07_BUILD_STRATEGY.md
│   └── 08_ETHICAL_GUIDELINES.md
│
├── 📁 docker/                       # Containerization
│   ├── Dockerfile.frontend
│   ├── Dockerfile.backend
│   ├── Dockerfile.ai
│   └── docker-compose.yml
│
├── 📁 tests/                        # Test Suites
│   ├── 📁 frontend/
│   ├── 📁 backend/
│   └── 📁 ai-engine/
│
├── .gitignore
├── README.md
└── LICENSE
```

---

## Module Descriptions

### 1. Frontend Module (`/frontend`)

| Component | Purpose |
|-----------|---------|
| `components/detection/` | Media upload, drag-drop, progress tracking |
| `components/forensics/` | Heatmap overlays, landmark visualization |
| `components/analytics/` | Charts, graphs, statistical displays |
| `pages/DetectionPage` | Main detection workflow interface |
| `pages/AnalyticsPage` | Dashboard with metrics and trends |
| `pages/EthicsPage` | Educational content about deepfakes |

### 2. Backend Module (`/backend`)

| Component | Purpose |
|-----------|---------|
| `controllers/` | Request handling and response formatting |
| `services/` | Business logic, AI service communication |
| `models/` | MongoDB schemas and data validation |
| `middleware/` | Auth, file upload, rate limiting |

### 3. AI Engine Module (`/ai-engine`)

| Component | Purpose |
|-----------|---------|
| `detection/` | Model inference for classification |
| `forensics/` | Facial analysis, blink detection, artifacts |
| `explainability/` | Grad-CAM, LIME, explanation generation |
| `ensemble/` | Multi-model voting and fusion |
| `training/` | Scripts for model training and evaluation |

---

*Document Version: 1.0 | Created: 2026-02-07*
