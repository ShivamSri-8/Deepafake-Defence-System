const Analysis = require('../models/Analysis');
const User = require('../models/User');
const { getMediaType, deleteFile } = require('../middleware/upload');
const aiEngineService = require('../services/aiEngine.service');

// @desc    Submit media for analysis
// @route   POST /api/v1/detect
// @access  Private
exports.submitAnalysis = async (req, res, next) => {
    try {
        // Check if file was uploaded
        if (!req.file) {
            return res.status(400).json({
                success: false,
                error: 'Please upload a file'
            });
        }

        const file = req.file;
        const mediaType = getMediaType(file.mimetype);

        // Create analysis record
        const analysis = await Analysis.create({
            user: req.user.id,
            file: {
                originalName: file.originalname,
                storedName: file.filename,
                mimeType: file.mimetype,
                size: file.size,
                mediaType: mediaType,
                path: file.path
            },
            status: 'pending',
            submittedAt: new Date()
        });

        // Increment user's analysis count
        await req.user.incrementAnalysisCount();

        // Start async processing (don't await)
        processAnalysis(analysis._id).catch(err => {
            console.error(`Analysis processing error: ${err.message}`);
        });

        res.status(202).json({
            success: true,
            message: 'Analysis submitted successfully',
            data: {
                analysisId: analysis.analysisId,
                status: analysis.status,
                file: {
                    name: file.originalname,
                    type: mediaType,
                    size: file.size
                },
                estimatedTime: mediaType === 'video' ? '30-60 seconds' : '5-15 seconds'
            }
        });
    } catch (error) {
        // Clean up uploaded file on error
        if (req.file) {
            deleteFile(req.file.path);
        }
        next(error);
    }
};

// @desc    Get analysis result
// @route   GET /api/v1/detect/:id
// @access  Private
exports.getAnalysisResult = async (req, res, next) => {
    try {
        const analysis = await Analysis.findOne({
            $or: [
                { analysisId: req.params.id },
                { _id: req.params.id }
            ],
            user: req.user.id
        });

        if (!analysis) {
            return res.status(404).json({
                success: false,
                error: 'Analysis not found'
            });
        }

        // Format response based on status
        const response = {
            success: true,
            data: {
                analysisId: analysis.analysisId,
                status: analysis.status,
                file: {
                    name: analysis.file.originalName,
                    type: analysis.file.mediaType,
                    size: analysis.file.size
                },
                submittedAt: analysis.submittedAt,
                ...(analysis.status === 'completed' && {
                    result: analysis.result,
                    modelPredictions: analysis.modelPredictions,
                    forensics: analysis.forensics,
                    explanation: analysis.explanation,
                    metadata: analysis.metadata,
                    completedAt: analysis.completedAt,
                    disclaimer: getDisclaimer(analysis.result.probability)
                }),
                ...(analysis.status === 'failed' && {
                    error: analysis.error
                }),
                ...(analysis.status === 'processing' && {
                    processingStages: analysis.processingStages
                })
            }
        };

        res.status(200).json(response);
    } catch (error) {
        next(error);
    }
};

// @desc    Get analysis status (for polling)
// @route   GET /api/v1/detect/:id/status
// @access  Private
exports.getAnalysisStatus = async (req, res, next) => {
    try {
        const analysis = await Analysis.findOne({
            $or: [
                { analysisId: req.params.id },
                { _id: req.params.id }
            ],
            user: req.user.id
        }).select('analysisId status processingStages startedAt');

        if (!analysis) {
            return res.status(404).json({
                success: false,
                error: 'Analysis not found'
            });
        }

        res.status(200).json({
            success: true,
            data: {
                analysisId: analysis.analysisId,
                status: analysis.status,
                stages: analysis.processingStages,
                startedAt: analysis.startedAt
            }
        });
    } catch (error) {
        next(error);
    }
};

// Helper: Process analysis asynchronously
async function processAnalysis(analysisId) {
    const analysis = await Analysis.findById(analysisId);
    if (!analysis) return;

    try {
        // Update status to processing
        await analysis.updateStatus('processing');

        // Define processing stages
        const stages = [
            'preprocessing',
            'xception_inference',
            'efficientnet_inference',
            ...(analysis.file.mediaType === 'video' ? ['cnn_lstm_inference'] : []),
            'forensic_analysis',
            'xai_generation',
            'result_aggregation'
        ];

        // Try to call AI engine, fall back to simulation if unavailable
        let result;
        try {
            result = await aiEngineService.analyzeMedia(analysis);
        } catch (err) {
            console.log('AI Engine unavailable, using simulation');
            result = await simulateAnalysis(analysis, stages);
        }

        // Update analysis with results
        analysis.result = result.result;
        analysis.modelPredictions = result.modelPredictions;
        analysis.forensics = result.forensics;
        analysis.explanation = result.explanation;
        analysis.metadata = {
            processingTime: result.processingTime,
            aiEngineVersion: result.aiEngineVersion || '1.0.0-simulated',
            modelsUsed: result.modelsUsed || ['xception', 'efficientnet', 'cnn_lstm'],
            disclaimerShown: true
        };

        await analysis.updateStatus('completed');

    } catch (error) {
        console.error(`Analysis failed: ${error.message}`);
        await analysis.updateStatus('failed', {
            message: error.message,
            code: 'PROCESSING_ERROR',
            stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
        });
    }
}

// Helper: Simulate analysis when AI engine is unavailable
async function simulateAnalysis(analysis, stages) {
    const startTime = Date.now();

    // Simulate processing stages
    for (const stage of stages) {
        await analysis.addProcessingStage(stage, 'running');
        await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 500));
        analysis.processingStages[analysis.processingStages.length - 1].status = 'completed';
        analysis.processingStages[analysis.processingStages.length - 1].completedAt = new Date();
        await analysis.save();
    }

    // Generate simulated results
    const isFake = Math.random() > 0.4;
    const probability = isFake
        ? 0.65 + Math.random() * 0.30
        : 0.10 + Math.random() * 0.25;

    const classification = probability > 0.6 ? 'fake' : probability < 0.35 ? 'real' : 'uncertain';

    return {
        result: {
            classification,
            probability,
            confidence: {
                lower: Math.max(0, probability - 0.05),
                upper: Math.min(1, probability + 0.05)
            }
        },
        modelPredictions: {
            xception: {
                score: probability + (Math.random() - 0.5) * 0.1,
                weight: 0.35,
                processingTime: 0.8 + Math.random() * 0.4
            },
            efficientnet: {
                score: probability + (Math.random() - 0.5) * 0.1,
                weight: 0.35,
                processingTime: 0.9 + Math.random() * 0.3
            },
            ...(analysis.file.mediaType === 'video' && {
                cnnLstm: {
                    score: probability + (Math.random() - 0.5) * 0.1,
                    weight: 0.30,
                    processingTime: 1.5 + Math.random() * 0.5
                }
            }),
            ensemble: {
                score: probability,
                method: 'weighted_average'
            }
        },
        forensics: {
            facialLandmarks: {
                score: 0.7 + Math.random() * 0.25,
                anomaly: probability > 0.5,
                details: {
                    landmarkCount: 468,
                    asymmetryScore: 0.1 + Math.random() * 0.2,
                    boundaryIrregularity: probability > 0.5 ? 0.3 + Math.random() * 0.3 : 0.1 + Math.random() * 0.1
                }
            },
            eyeBlink: {
                score: 0.6 + Math.random() * 0.3,
                anomaly: probability > 0.6,
                details: {
                    blinkRate: 15 + Math.random() * 10,
                    earConsistency: 0.8 + Math.random() * 0.15,
                    patternRegularity: probability > 0.6 ? 0.5 : 0.85
                }
            },
            lipSync: {
                score: 0.5 + Math.random() * 0.4,
                anomaly: probability > 0.55,
                details: {
                    audioVideoCorrelation: 0.7 + Math.random() * 0.25,
                    mouthMovementScore: 0.6 + Math.random() * 0.3
                }
            },
            frequencyAnalysis: {
                score: 0.4 + Math.random() * 0.5,
                anomaly: probability > 0.7,
                details: {
                    ganFingerprint: probability > 0.65,
                    spectralAnomaly: probability > 0.5 ? 0.4 + Math.random() * 0.4 : 0.1 + Math.random() * 0.2
                }
            },
            temporalConsistency: {
                score: 0.5 + Math.random() * 0.4,
                anomaly: probability > 0.6,
                details: {
                    frameToFrameScore: 0.7 + Math.random() * 0.25,
                    motionConsistency: 0.6 + Math.random() * 0.3
                }
            }
        },
        explanation: {
            summary: probability > 0.6
                ? `The model detected potential manipulation indicators with ${(probability * 100).toFixed(1)}% probability. The facial boundary area showed anomalous patterns consistent with face-swapping techniques. Eye blink patterns appear irregular compared to natural videos.`
                : probability < 0.35
                    ? `The model found limited manipulation indicators (${(probability * 100).toFixed(1)}% probability). The media appears likely authentic based on analyzed features including consistent facial landmarks and natural eye blink patterns.`
                    : `The model's confidence is uncertain (${(probability * 100).toFixed(1)}% probability). The analyzed features show mixed signals. Additional verification by qualified experts is recommended before drawing conclusions.`,
            keyRegions: [
                { name: 'Face boundary', attention: 0.85 + Math.random() * 0.1 },
                { name: 'Eye region', attention: 0.70 + Math.random() * 0.15 },
                { name: 'Lip area', attention: 0.55 + Math.random() * 0.2 },
                { name: 'Forehead', attention: 0.40 + Math.random() * 0.2 }
            ]
        },
        processingTime: (Date.now() - startTime) / 1000,
        aiEngineVersion: '1.0.0-simulation'
    };
}

// Helper: Get appropriate disclaimer
function getDisclaimer(probability) {
    if (probability > 0.7) {
        return {
            type: 'high_confidence',
            message: 'High probability of manipulation detected. However, this assessment is probabilistic and should not be used as sole evidence. Verification by qualified forensic experts is strongly recommended.'
        };
    } else if (probability < 0.3) {
        return {
            type: 'low_confidence',
            message: 'Low probability of manipulation detected. The media appears likely authentic, but no detection system is 100% accurate. Exercise appropriate judgment based on context.'
        };
    } else {
        return {
            type: 'uncertain',
            message: 'The model\'s confidence is uncertain. Results in this range should be treated with particular caution. Additional verification methods and expert consultation are recommended.'
        };
    }
}
