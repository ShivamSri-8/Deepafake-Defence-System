# API Documentation

---

## 1. API Overview

| Property | Value |
|----------|-------|
| Base URL | `http://localhost:8080/api/v1` |
| Format | JSON |
| Authentication | JWT Bearer Token |
| Rate Limit | 100 requests/minute |

---

## 2. Endpoints

### 2.1 Detection Endpoints

#### POST `/detect`
Submit media for deepfake analysis.

**Request:**
```http
POST /api/v1/detect
Content-Type: multipart/form-data
Authorization: Bearer <token>

{
  "file": <binary>,
  "options": {
    "runForensics": true,
    "generateExplanation": true,
    "priority": "normal"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "analysisId": "a3f8c7d2-1234-5678-abcd-ef9012345678",
    "timestamp": "2026-02-07T13:20:00Z",
    "processingTime": 2.34,
    
    "classification": {
      "label": "potentially_manipulated",
      "probability": 0.783,
      "confidence": {
        "lower": 0.731,
        "upper": 0.835
      }
    },
    
    "modelPredictions": {
      "xception": { "probability": 0.812, "weight": 0.35 },
      "efficientnet": { "probability": 0.756, "weight": 0.35 },
      "ensemble": { "probability": 0.783, "method": "weighted_average" }
    },
    
    "forensicAnalysis": {
      "overallScore": 0.67,
      "indicators": [
        { "feature": "facial_landmarks", "status": "anomaly", "score": 0.72 },
        { "feature": "eye_blink", "status": "anomaly", "score": 0.81 }
      ]
    },
    
    "explanation": {
      "summary": "The model detected potential manipulation...",
      "visualizations": {
        "gradcam": "/api/v1/visualizations/a3f8c7d2/gradcam.png",
        "overlay": "/api/v1/visualizations/a3f8c7d2/overlay.png"
      }
    },
    
    "disclaimer": "This is a probabilistic assessment..."
  }
}
```

#### GET `/detect/:id`
Retrieve analysis result by ID.

**Response:**
```json
{
  "success": true,
  "data": {
    "analysisId": "a3f8c7d2-1234-5678-abcd-ef9012345678",
    "status": "completed",
    "result": { ... }
  }
}
```

---

### 2.2 History Endpoints

#### GET `/history`
List analysis history with pagination.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | number | 1 | Page number |
| limit | number | 20 | Items per page |
| filter | string | all | Filter: all, real, fake |
| sortBy | string | createdAt | Sort field |
| order | string | desc | Sort order: asc, desc |

**Response:**
```json
{
  "success": true,
  "data": {
    "items": [ ... ],
    "pagination": {
      "page": 1,
      "limit": 20,
      "total": 156,
      "pages": 8
    }
  }
}
```

---

### 2.3 Analytics Endpoints

#### GET `/analytics`
Get aggregate statistics.

**Response:**
```json
{
  "success": true,
  "data": {
    "summary": {
      "totalAnalyses": 1542,
      "realCount": 823,
      "fakeCount": 612,
      "uncertainCount": 107,
      "averageConfidence": 0.76
    },
    "confidenceDistribution": [
      { "range": "0-20", "count": 45 },
      { "range": "20-40", "count": 89 },
      { "range": "40-60", "count": 234 },
      { "range": "60-80", "count": 567 },
      { "range": "80-100", "count": 607 }
    ],
    "mediaTypes": {
      "image": 1123,
      "video": 419
    }
  }
}
```

#### GET `/analytics/trends`
Get time-based trend data.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| period | string | week | Period: day, week, month, year |
| metric | string | count | Metric: count, confidence, accuracy |

**Response:**
```json
{
  "success": true,
  "data": {
    "period": "week",
    "dataPoints": [
      { "date": "2026-02-01", "total": 45, "real": 23, "fake": 22 },
      { "date": "2026-02-02", "total": 52, "real": 28, "fake": 24 }
    ]
  }
}
```

---

### 2.4 Model Metrics Endpoints

#### GET `/model/metrics`
Get model performance metrics (Admin only).

**Response:**
```json
{
  "success": true,
  "data": {
    "models": [
      {
        "name": "xception",
        "version": "1.2.0",
        "metrics": {
          "accuracy": 0.943,
          "precision": 0.931,
          "recall": 0.956,
          "f1Score": 0.943,
          "aucRoc": 0.978
        },
        "confusionMatrix": [[456, 32], [21, 491]],
        "evaluatedOn": "FaceForensics++",
        "lastUpdated": "2026-01-15T10:00:00Z"
      }
    ]
  }
}
```

---

### 2.5 Ethics Endpoints

#### GET `/ethics/guidelines`
Get ethical usage guidelines.

**Response:**
```json
{
  "success": true,
  "data": {
    "guidelines": [
      {
        "title": "Responsible Usage",
        "content": "This system is designed as a decision-support tool..."
      }
    ],
    "disclaimer": "...",
    "resources": [
      { "title": "Understanding Deepfakes", "url": "..." }
    ]
  }
}
```

---

## 3. Error Responses

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid file format",
    "details": {
      "field": "file",
      "allowed": ["image/jpeg", "image/png", "video/mp4"]
    }
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| VALIDATION_ERROR | 400 | Invalid input data |
| UNAUTHORIZED | 401 | Missing or invalid token |
| FORBIDDEN | 403 | Insufficient permissions |
| NOT_FOUND | 404 | Resource not found |
| RATE_LIMITED | 429 | Too many requests |
| PROCESSING_ERROR | 500 | Internal processing error |
| MODEL_ERROR | 503 | AI model unavailable |

---

## 4. WebSocket Events

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8080/ws');
ws.send(JSON.stringify({ type: 'subscribe', analysisId: 'xxx' }));
```

### Events

| Event | Description |
|-------|-------------|
| `progress` | Processing progress update |
| `complete` | Analysis complete |
| `error` | Processing error |

```json
{
  "type": "progress",
  "data": {
    "analysisId": "xxx",
    "stage": "forensics",
    "progress": 65,
    "message": "Analyzing facial landmarks..."
  }
}
```

---

*Document Version: 1.0 | Created: 2026-02-07*
