const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const {
    computeAgreementScore,
    computeTrustScore,
    computeTemporalVariance,
    computeConfidenceLevel,
} = require('../utils/trustScore');

const AI_ENGINE_URL = process.env.AI_ENGINE_URL || 'http://localhost:8000';

/**
 * AI Engine Service
 * Handles communication with the Python-based AI inference engine (FastAPI)
 */
class AIEngineService {
    constructor() {
        this.client = axios.create({
            baseURL: AI_ENGINE_URL,
            timeout: 120000, // 2 minutes timeout for large files
        });
        this.simulationMode = false;
    }

    /**
     * Check if AI Engine is available
     */
    async healthCheck() {
        try {
            const response = await this.client.get('/health');
            this.simulationMode = !response.data.models_loaded;
            return response.data.status === 'ok' || response.data.status === 'healthy';
        } catch (error) {
            console.warn('AI Engine not available, using simulation mode:', error.message);
            this.simulationMode = true;
            return false;
        }
    }

    /**
     * Get detailed health information
     */
    async getDetailedHealth() {
        try {
            const response = await this.client.get('/health/detailed');
            return response.data;
        } catch (error) {
            return { status: 'unavailable', error: error.message };
        }
    }

    /**
     * Analyze media file for deepfakes
     * @param {Object} analysis - Analysis document with file information
     */
    async analyzeMedia(analysis) {
        const filePath = analysis.file.path;

        if (!fs.existsSync(filePath)) {
            throw new Error('Media file not found');
        }

        // Check if AI Engine is available
        const isAvailable = await this.healthCheck();

        if (!isAvailable) {
            console.log('AI Engine unavailable - using simulation mode');
            return this.simulateAnalysis(analysis);
        }

        const formData = new FormData();
        formData.append('file', fs.createReadStream(filePath));

        try {
            // Call detection endpoint
            const detectResponse = await this.client.post('/api/v1/detect', formData, {
                headers: {
                    ...formData.getHeaders(),
                },
                maxContentLength: Infinity,
                maxBodyLength: Infinity,
            });

            return this.formatResponse(detectResponse.data);
        } catch (error) {
            if (error.code === 'ECONNREFUSED') {
                console.warn('AI Engine connection refused - using simulation');
                return this.simulateAnalysis(analysis);
            }
            throw new Error(`AI Engine error: ${error.response?.data?.detail || error.message}`);
        }
    }

    /**
     * Get forensic analysis for media
     */
    async getForensics(filePath) {
        if (!fs.existsSync(filePath)) {
            throw new Error('Media file not found');
        }

        const formData = new FormData();
        formData.append('file', fs.createReadStream(filePath));

        try {
            const response = await this.client.post('/api/v1/forensics/analyze', formData, {
                headers: formData.getHeaders(),
            });
            return response.data;
        } catch (error) {
            console.warn('Forensics analysis failed:', error.message);
            return this.simulateForensics();
        }
    }

    /**
     * Get explainability (XAI) analysis
     */
    async getExplanation(filePath) {
        if (!fs.existsSync(filePath)) {
            throw new Error('Media file not found');
        }

        const formData = new FormData();
        formData.append('file', fs.createReadStream(filePath));
        formData.append('include_lime', 'false'); // Disable LIME by default (too slow on CPU)

        try {
            const response = await this.client.post('/api/v1/explain', formData, {
                headers: formData.getHeaders(),
            });
            return response.data;
        } catch (error) {
            console.warn('XAI analysis failed:', error.message);
            return this.simulateExplanation();
        }
    }

    /**
     * Simulate analysis when AI Engine is unavailable
     */
    simulateAnalysis(analysis) {
        const probability = Math.random() * 0.6 + 0.2; // 0.2 to 0.8
        const classification = probability >= 0.60 ? 'fake' : probability < 0.40 ? 'real' : 'uncertain';

        // Trust-aware fields
        const modelScores = [
            probability + (Math.random() - 0.5) * 0.1,
            probability + (Math.random() - 0.5) * 0.1,
        ];
        const confidence = 0.87;
        const agreement = computeAgreementScore(modelScores);
        const trustScore = computeTrustScore(confidence, agreement);
        const confidenceLevel = computeConfidenceLevel(confidence * 100);

        return {
            result: {
                classification: classification,
                probability: probability,
                confidence: {
                    lower: Math.max(0, probability - 0.1),
                    upper: Math.min(1, probability + 0.1)
                }
            },
            modelPredictions: {
                xception: { probability: modelScores[0], confidence: 0.85 },
                efficientnet: { probability: modelScores[1], confidence: 0.88 },
                cnnLstm: null,
                ensemble: { probability: probability, confidence: 0.87 }
            },
            forensics: this.simulateForensics(),
            explanation: this.simulateExplanation(),
            processingTime: Math.random() * 2000 + 500,
            aiEngineVersion: '1.0.0-simulation',
            modelsUsed: ['simulation'],
            isSimulated: true,
            disclaimer: 'This is a simulated result. AI Engine is not available.',
            // Trust-aware outputs
            trustScore: trustScore,
            temporalVariance: null,
            temporalLabel: null,
            confidenceLevel: confidenceLevel,
        };
    }

    /**
     * Simulate forensics data
     */
    simulateForensics() {
        return {
            facialLandmarks: {
                score: Math.random() * 0.3 + 0.6,
                anomaly: Math.random() > 0.7,
                details: { regions: ['eyes', 'mouth', 'nose'] }
            },
            eyeBlink: {
                score: Math.random() * 0.3 + 0.6,
                anomaly: Math.random() > 0.8,
                details: { blinkRate: 15 + Math.random() * 10 }
            },
            lipSync: {
                score: Math.random() * 0.3 + 0.6,
                anomaly: Math.random() > 0.8,
                details: {}
            },
            frequencyAnalysis: {
                score: Math.random() * 0.4 + 0.5,
                anomaly: Math.random() > 0.6,
                details: { artifactsDetected: Math.random() > 0.5 }
            },
            temporalConsistency: {
                score: Math.random() * 0.3 + 0.6,
                anomaly: Math.random() > 0.7,
                details: {}
            }
        };
    }

    /**
     * Simulate explanation data
     */
    simulateExplanation() {
        return {
            summary: 'This is a simulated analysis. The AI Engine is currently unavailable. For accurate results, please ensure the AI Engine is running.',
            keyRegions: [
                { name: 'eyes', importance: 0.85, finding: 'Simulated attention region' },
                { name: 'mouth', importance: 0.72, finding: 'Simulated attention region' },
                { name: 'nose_bridge', importance: 0.65, finding: 'Simulated attention region' }
            ],
            gradcamPath: null,
            limePath: null
        };
    }

    /**
     * Get analysis progress
     * @param {string} analysisId - Analysis ID to check
     */
    async getProgress(analysisId) {
        try {
            const response = await this.client.get(`/api/v1/analyze/${analysisId}/progress`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get progress: ${error.message}`);
        }
    }

    /**
     * Format AI Engine response to match our schema.
     *
     * The AI engine returns a DetectionResponse:
     * {
     *   analysis_id, status, file_type, filename, timestamp,
     *   result: DetectionResult {
     *     is_fake, fake_probability, confidence,
     *     confidence_interval: { lower, upper },
     *     risk_level, model_predictions: [ { model_name, fake_probability, confidence, weight } ],
     *     trust_score, temporal_variance, temporal_label, confidence_level,
     *     notes: []
     *   },
     *   disclaimer
     * }
     */
    formatResponse(data) {
        // The core detection result lives under data.result
        const det = data.result || {};

        // Normalise classification to 'real' / 'fake' / 'uncertain'
        const probability = det.fake_probability ?? 0.5;
        let classification = 'uncertain';
        if (probability >= 0.60) classification = 'fake';
        else if (probability < 0.40) classification = 'real';

        // model_predictions is an ARRAY of { model_name, fake_probability, confidence, weight }
        const modelPredictions = {};
        const rawPreds = Array.isArray(det.model_predictions) ? det.model_predictions : [];
        rawPreds.forEach(pred => {
            const key = this._normaliseModelKey(pred.model_name || '');
            modelPredictions[key] = {
                score: pred.fake_probability ?? 0,
                weight: pred.weight ?? 0,
                confidence: pred.confidence ?? 0,
            };
        });

        return {
            result: {
                classification,
                probability,
                confidence: {
                    lower: det.confidence_interval?.lower ?? Math.max(0, probability - 0.05),
                    upper: det.confidence_interval?.upper ?? Math.min(1, probability + 0.05),
                }
            },
            modelPredictions,
            forensics: this.formatForensics(data.forensics),
            explanation: {
                summary: data.explanation?.summary || '',
                keyRegions: data.explanation?.key_regions || [],
                gradcamPath: data.explanation?.gradcam_path || null,
                limePath: data.explanation?.lime_path || null
            },
            processingTime: data.processing_time || 0,
            aiEngineVersion: data.version || '1.0.0',
            modelsUsed: data.models_used || ['xception', 'efficientnet'],
            isSimulated: false,
            // Trust-aware fields (forwarded from AI engine DetectionResult)
            trustScore: det.trust_score ?? null,
            temporalVariance: det.temporal_variance ?? null,
            temporalLabel: det.temporal_label ?? null,
            confidenceLevel: det.confidence_level ?? null,
        };
    }

    /**
     * Normalise a model_name string into a camelCase key
     */
    _normaliseModelKey(name) {
        const n = name.toLowerCase().replace(/[^a-z0-9]/g, '');
        if (n.includes('cnn') && n.includes('lstm')) return 'cnnLstm';
        if (n.includes('efficient')) return 'efficientnet';
        if (n.includes('xception')) return 'xception';
        if (n.includes('resnet')) return 'resnet50';
        if (n.includes('ensemble')) return 'ensemble';
        if (n.includes('frame')) return 'frameAnalysis';
        return n;
    }

    /**
     * Format forensics data
     */
    formatForensics(forensics) {
        if (!forensics) return {};

        return {
            facialLandmarks: {
                score: forensics.facial_landmarks?.score || 0,
                anomaly: forensics.facial_landmarks?.anomaly || false,
                details: forensics.facial_landmarks?.details || {}
            },
            eyeBlink: {
                score: forensics.eye_blink?.score || 0,
                anomaly: forensics.eye_blink?.anomaly || false,
                details: forensics.eye_blink?.details || {}
            },
            lipSync: {
                score: forensics.lip_sync?.score || 0,
                anomaly: forensics.lip_sync?.anomaly || false,
                details: forensics.lip_sync?.details || {}
            },
            frequencyAnalysis: {
                score: forensics.frequency_analysis?.score || 0,
                anomaly: forensics.frequency_analysis?.anomaly || false,
                details: forensics.frequency_analysis?.details || {}
            },
            temporalConsistency: {
                score: forensics.temporal_consistency?.score || 0,
                anomaly: forensics.temporal_consistency?.anomaly || false,
                details: forensics.temporal_consistency?.details || {}
            }
        };
    }
}

module.exports = new AIEngineService();
