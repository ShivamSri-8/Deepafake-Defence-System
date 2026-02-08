# Database Schema Documentation

---

## 1. Overview

| Database | MongoDB 6.0+ |
|----------|--------------|
| ODM | Mongoose (Node.js) |
| Naming Convention | camelCase fields, PascalCase collections |

---

## 2. Collections

### 2.1 Analyses Collection

Stores all detection analysis results.

```javascript
// analyses collection schema
const AnalysisSchema = {
  _id: ObjectId,
  analysisId: { type: String, required: true, unique: true, index: true },
  createdAt: { type: Date, default: Date.now, index: true },
  updatedAt: { type: Date, default: Date.now },
  
  // Input metadata
  input: {
    type: { type: String, enum: ['image', 'video'], required: true },
    originalName: { type: String, required: true },
    mimeType: { type: String, required: true },
    fileSize: { type: Number, required: true },
    dimensions: {
      width: Number,
      height: Number
    },
    duration: Number,        // For videos (seconds)
    frameCount: Number,      // For videos
    storagePath: String,
    checksum: String         // SHA-256 hash
  },
  
  // Detection results
  detection: {
    classification: { 
      type: String, 
      enum: ['real', 'fake', 'uncertain'],
      required: true 
    },
    probability: { type: Number, min: 0, max: 1, required: true },
    confidenceInterval: {
      lower: { type: Number, min: 0, max: 1 },
      upper: { type: Number, min: 0, max: 1 }
    },
    modelPredictions: {
      xception: { 
        score: Number, 
        processingTime: Number  // milliseconds
      },
      efficientnet: { 
        score: Number, 
        processingTime: Number 
      },
      cnnLstm: { 
        score: Number, 
        processingTime: Number 
      }
    },
    ensembleMethod: { type: String, default: 'weighted_average' },
    totalProcessingTime: Number  // milliseconds
  },
  
  // Forensic analysis
  forensics: {
    facialLandmarks: {
      symmetryScore: { type: Number, min: 0, max: 1 },
      anomalyDetected: Boolean,
      landmarkCount: Number,
      details: Schema.Types.Mixed
    },
    eyeAnalysis: {
      blinkRate: Number,  // blinks per second
      earValues: [Number],
      anomalyDetected: Boolean,
      blinkFrames: [Number]
    },
    lipSync: {
      coherenceScore: { type: Number, min: 0, max: 1 },
      anomalyDetected: Boolean
    },
    frequencyAnalysis: {
      ganFingerprint: Boolean,
      artifactScore: { type: Number, min: 0, max: 1 }
    },
    temporalConsistency: {
      flowScore: { type: Number, min: 0, max: 1 },
      flickerDetected: Boolean,
      motionAnomaly: Boolean
    },
    overallScore: { type: Number, min: 0, max: 1 }
  },
  
  // Explainability
  explanation: {
    gradcamPath: String,
    limePath: String,
    overlayPath: String,
    textualSummary: String,
    detailedExplanations: [String],
    keyRegions: [{
      name: String,
      attentionScore: Number,
      coordinates: {
        x: Number,
        y: Number,
        w: Number,
        h: Number
      }
    }],
    confidenceLevel: { type: String, enum: ['high', 'moderate', 'low'] }
  },
  
  // Metadata
  metadata: {
    clientIpHash: String,    // Hashed for privacy
    userAgent: String,
    processingNode: String,
    modelVersions: {
      xception: String,
      efficientnet: String,
      cnnLstm: String
    },
    apiVersion: String
  },
  
  // User feedback (optional)
  feedback: {
    userReported: Boolean,
    reportedLabel: { type: String, enum: ['real', 'fake'] },
    feedbackText: String,
    feedbackDate: Date
  }
};

// Indexes
AnalysisSchema.index({ createdAt: -1 });
AnalysisSchema.index({ 'detection.classification': 1 });
AnalysisSchema.index({ 'detection.probability': 1 });
```

---

### 2.2 Analytics Collection

Stores daily aggregated statistics.

```javascript
// analytics collection schema
const AnalyticsSchema = {
  _id: ObjectId,
  date: { type: Date, required: true, unique: true, index: true },
  
  metrics: {
    totalAnalyses: { type: Number, default: 0 },
    realCount: { type: Number, default: 0 },
    fakeCount: { type: Number, default: 0 },
    uncertainCount: { type: Number, default: 0 },
    averageConfidence: { type: Number, default: 0 },
    averageProcessingTime: { type: Number, default: 0 },
    imageCount: { type: Number, default: 0 },
    videoCount: { type: Number, default: 0 }
  },
  
  confidenceDistribution: [{
    range: String,      // e.g., "0-10", "10-20"
    count: Number
  }],
  
  processingTimeDistribution: [{
    range: String,      // e.g., "0-1s", "1-2s"
    count: Number
  }],
  
  hourlyBreakdown: [{
    hour: Number,       // 0-23
    count: Number
  }],
  
  forensicAnomalies: {
    facialLandmarks: Number,
    eyeBlink: Number,
    lipSync: Number,
    frequency: Number,
    temporal: Number
  }
};
```

---

### 2.3 ModelMetrics Collection

Stores model performance evaluations.

```javascript
// modelMetrics collection schema
const ModelMetricsSchema = {
  _id: ObjectId,
  modelName: { type: String, required: true, index: true },
  version: { type: String, required: true },
  evaluatedAt: { type: Date, default: Date.now },
  
  dataset: {
    name: String,           // e.g., "FaceForensics++"
    version: String,
    splitType: String,      // "test", "validation"
    sampleCount: Number
  },
  
  metrics: {
    accuracy: { type: Number, min: 0, max: 1 },
    precision: { type: Number, min: 0, max: 1 },
    recall: { type: Number, min: 0, max: 1 },
    f1Score: { type: Number, min: 0, max: 1 },
    aucRoc: { type: Number, min: 0, max: 1 },
    aucPr: { type: Number, min: 0, max: 1 },
    logLoss: Number,
    brierScore: Number
  },
  
  confusionMatrix: {
    truePositive: Number,
    trueNegative: Number,
    falsePositive: Number,
    falseNegative: Number,
    matrix: [[Number]]      // 2x2 matrix
  },
  
  thresholdAnalysis: [{
    threshold: Number,
    precision: Number,
    recall: Number,
    f1: Number
  }],
  
  perClassMetrics: {
    real: {
      precision: Number,
      recall: Number,
      f1: Number,
      support: Number
    },
    fake: {
      precision: Number,
      recall: Number,
      f1: Number,
      support: Number
    }
  },
  
  trainingInfo: {
    epochs: Number,
    batchSize: Number,
    learningRate: Number,
    optimizer: String,
    trainingTime: Number    // hours
  }
};

// Compound index for latest version queries
ModelMetricsSchema.index({ modelName: 1, evaluatedAt: -1 });
```

---

### 2.4 Users Collection (Optional)

For admin/research access control.

```javascript
// users collection schema
const UserSchema = {
  _id: ObjectId,
  email: { type: String, required: true, unique: true },
  passwordHash: { type: String, required: true },
  role: { 
    type: String, 
    enum: ['user', 'researcher', 'admin'], 
    default: 'user' 
  },
  createdAt: { type: Date, default: Date.now },
  lastLogin: Date,
  isActive: { type: Boolean, default: true },
  
  preferences: {
    darkMode: Boolean,
    emailNotifications: Boolean
  },
  
  apiKey: {
    key: String,
    createdAt: Date,
    lastUsed: Date
  }
};
```

---

## 3. Relationships

```
                    ┌─────────────────┐
                    │     Users       │
                    │  (Optional)     │
                    └────────┬────────┘
                             │ 1:N
                             ▼
┌─────────────────┐    ┌─────────────┐    ┌─────────────────┐
│  ModelMetrics   │    │  Analyses   │    │   Analytics     │
│                 │    │             │    │   (Daily)       │
│  - Performance  │    │  - Results  │───▶│  - Aggregates   │
│  - Evaluation   │    │  - Reports  │    │  - Trends       │
└─────────────────┘    └─────────────┘    └─────────────────┘
```

---

## 4. Mongoose Models (Node.js)

```javascript
// models/Analysis.js
const mongoose = require('mongoose');

const analysisSchema = new mongoose.Schema({
  analysisId: { type: String, required: true, unique: true },
  input: { /* as defined above */ },
  detection: { /* as defined above */ },
  forensics: { /* as defined above */ },
  explanation: { /* as defined above */ },
  metadata: { /* as defined above */ },
  feedback: { /* as defined above */ }
}, {
  timestamps: true,
  collection: 'analyses'
});

// Virtual for classification label
analysisSchema.virtual('classificationLabel').get(function() {
  if (this.detection.probability > 0.7) return 'likely_fake';
  if (this.detection.probability < 0.3) return 'likely_real';
  return 'uncertain';
});

// Method to generate summary
analysisSchema.methods.getSummary = function() {
  return {
    id: this.analysisId,
    classification: this.detection.classification,
    probability: this.detection.probability,
    confidence: this.detection.confidenceInterval,
    createdAt: this.createdAt
  };
};

module.exports = mongoose.model('Analysis', analysisSchema);
```

---

## 5. Sample Queries

```javascript
// Get recent analyses
db.analyses.find({})
  .sort({ createdAt: -1 })
  .limit(20);

// Get fake detections with high confidence
db.analyses.find({
  'detection.classification': 'fake',
  'detection.probability': { $gt: 0.8 }
});

// Aggregate daily statistics
db.analyses.aggregate([
  {
    $match: {
      createdAt: { 
        $gte: ISODate("2026-02-01"), 
        $lt: ISODate("2026-02-08") 
      }
    }
  },
  {
    $group: {
      _id: { $dateToString: { format: "%Y-%m-%d", date: "$createdAt" } },
      total: { $sum: 1 },
      avgConfidence: { $avg: "$detection.probability" },
      fakeCount: {
        $sum: { $cond: [{ $eq: ["$detection.classification", "fake"] }, 1, 0] }
      }
    }
  },
  { $sort: { _id: 1 } }
]);

// Get model performance comparison
db.modelMetrics.find({})
  .sort({ 'metrics.aucRoc': -1 });
```

---

*Document Version: 1.0 | Created: 2026-02-07*
