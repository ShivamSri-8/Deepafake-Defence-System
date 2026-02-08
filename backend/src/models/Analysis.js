const mongoose = require('mongoose');
const { v4: uuidv4 } = require('uuid');

const analysisSchema = new mongoose.Schema({
    analysisId: {
        type: String,
        unique: true,
        default: () => uuidv4()
    },
    user: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    // File Information
    file: {
        originalName: {
            type: String,
            required: true
        },
        storedName: {
            type: String,
            required: true
        },
        mimeType: {
            type: String,
            required: true
        },
        size: {
            type: Number,
            required: true
        },
        mediaType: {
            type: String,
            enum: ['image', 'video'],
            required: true
        },
        path: String,
        duration: Number, // For videos (seconds)
        frameCount: Number, // For videos
        dimensions: {
            width: Number,
            height: Number
        }
    },
    // Analysis Status
    status: {
        type: String,
        enum: ['pending', 'processing', 'completed', 'failed'],
        default: 'pending'
    },
    processingStages: [{
        stage: String,
        status: {
            type: String,
            enum: ['pending', 'running', 'completed', 'failed']
        },
        startedAt: Date,
        completedAt: Date,
        error: String
    }],
    // Detection Results
    result: {
        classification: {
            type: String,
            enum: ['real', 'fake', 'uncertain']
        },
        probability: {
            type: Number,
            min: 0,
            max: 1
        },
        confidence: {
            lower: Number,
            upper: Number
        }
    },
    // Model Predictions
    modelPredictions: {
        xception: {
            score: Number,
            weight: Number,
            processingTime: Number
        },
        efficientnet: {
            score: Number,
            weight: Number,
            processingTime: Number
        },
        cnnLstm: {
            score: Number,
            weight: Number,
            processingTime: Number
        },
        ensemble: {
            score: Number,
            method: String
        }
    },
    // Forensic Analysis
    forensics: {
        facialLandmarks: {
            score: Number,
            anomaly: Boolean,
            details: {
                landmarkCount: Number,
                asymmetryScore: Number,
                boundaryIrregularity: Number
            }
        },
        eyeBlink: {
            score: Number,
            anomaly: Boolean,
            details: {
                blinkRate: Number,
                earConsistency: Number,
                patternRegularity: Number
            }
        },
        lipSync: {
            score: Number,
            anomaly: Boolean,
            details: {
                audioVideoCorrelation: Number,
                mouthMovementScore: Number
            }
        },
        frequencyAnalysis: {
            score: Number,
            anomaly: Boolean,
            details: {
                ganFingerprint: Boolean,
                spectralAnomaly: Number
            }
        },
        temporalConsistency: {
            score: Number,
            anomaly: Boolean,
            details: {
                frameToFrameScore: Number,
                motionConsistency: Number
            }
        }
    },
    // Explainability
    explanation: {
        summary: String,
        keyRegions: [{
            name: String,
            attention: Number,
            coordinates: {
                x: Number,
                y: Number,
                width: Number,
                height: Number
            }
        }],
        gradcamPath: String,
        limePath: String
    },
    // Metadata
    metadata: {
        processingTime: Number, // Total processing time in seconds
        aiEngineVersion: String,
        modelsUsed: [String],
        disclaimerShown: {
            type: Boolean,
            default: false
        }
    },
    // Error Information (if failed)
    error: {
        message: String,
        code: String,
        stack: String
    },
    // Timestamps
    submittedAt: {
        type: Date,
        default: Date.now
    },
    startedAt: Date,
    completedAt: Date
}, {
    timestamps: true
});

// Indexes for efficient queries
analysisSchema.index({ user: 1, createdAt: -1 });
analysisSchema.index({ status: 1 });
analysisSchema.index({ 'result.classification': 1 });
analysisSchema.index({ 'file.mediaType': 1 });

// Virtual for processing duration
analysisSchema.virtual('processingDuration').get(function () {
    if (this.startedAt && this.completedAt) {
        return (this.completedAt - this.startedAt) / 1000; // seconds
    }
    return null;
});

// Instance method to update status
analysisSchema.methods.updateStatus = async function (status, error = null) {
    this.status = status;
    if (status === 'processing' && !this.startedAt) {
        this.startedAt = new Date();
    }
    if (status === 'completed' || status === 'failed') {
        this.completedAt = new Date();
    }
    if (error) {
        this.error = error;
    }
    await this.save();
};

// Instance method to add processing stage
analysisSchema.methods.addProcessingStage = async function (stage, status) {
    const stageEntry = {
        stage,
        status,
        startedAt: status === 'running' ? new Date() : undefined,
        completedAt: status === 'completed' ? new Date() : undefined
    };
    this.processingStages.push(stageEntry);
    await this.save();
};

// Static method to get user statistics
analysisSchema.statics.getUserStats = async function (userId) {
    const stats = await this.aggregate([
        { $match: { user: userId, status: 'completed' } },
        {
            $group: {
                _id: null,
                total: { $sum: 1 },
                realCount: {
                    $sum: { $cond: [{ $eq: ['$result.classification', 'real'] }, 1, 0] }
                },
                fakeCount: {
                    $sum: { $cond: [{ $eq: ['$result.classification', 'fake'] }, 1, 0] }
                },
                uncertainCount: {
                    $sum: { $cond: [{ $eq: ['$result.classification', 'uncertain'] }, 1, 0] }
                },
                avgConfidence: { $avg: '$result.probability' },
                avgProcessingTime: { $avg: '$metadata.processingTime' }
            }
        }
    ]);
    return stats[0] || { total: 0, realCount: 0, fakeCount: 0, uncertainCount: 0 };
};

module.exports = mongoose.model('Analysis', analysisSchema);
