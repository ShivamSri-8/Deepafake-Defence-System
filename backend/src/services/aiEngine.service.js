const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

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
            this.simulationMode = false;
            return response.data.status === 'healthy';
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
        const isFake = probability >= 0.5;

        return {
            result: {
                classification: isFake ? 'LIKELY_FAKE' : 'LIKELY_AUTHENTIC',
                probability: probability,
                confidence: {
                    lower: Math.max(0, probability - 0.1),
                    upper: Math.min(1, probability + 0.1)
                }
            },
            modelPredictions: {
                xception: { probability: probability + (Math.random() - 0.5) * 0.1, confidence: 0.85 },
                efficientnet: { probability: probability + (Math.random() - 0.5) * 0.1, confidence: 0.88 },
                cnnLstm: null,
                ensemble: { probability: probability, confidence: 0.87 }
            },
            forensics: this.simulateForensics(),
            explanation: this.simulateExplanation(),
            processingTime: Math.random() * 2000 + 500,
            aiEngineVersion: '1.0.0-simulation',
            modelsUsed: ['simulation'],
            isSimulated: true,
            disclaimer: 'This is a simulated result. AI Engine is not available.'
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
     * Format AI Engine response to match our schema
     */
    formatResponse(data) {
        return {
            result: {
                classification: data.classification,
                probability: data.probability,
                confidence: {
                    lower: data.confidence_interval?.lower || data.probability - 0.05,
                    upper: data.confidence_interval?.upper || data.probability + 0.05
                }
            },
            modelPredictions: {
                xception: data.model_predictions?.xception || null,
                efficientnet: data.model_predictions?.efficientnet || null,
                cnnLstm: data.model_predictions?.cnn_lstm || null,
                ensemble: data.model_predictions?.ensemble || null
            },
            forensics: this.formatForensics(data.forensics),
            explanation: {
                summary: data.explanation?.summary || '',
                keyRegions: data.explanation?.key_regions || [],
                gradcamPath: data.explanation?.gradcam_path || null,
                limePath: data.explanation?.lime_path || null
            },
            processingTime: data.processing_time || 0,
            aiEngineVersion: data.version || '1.0.0',
            modelsUsed: data.models_used || ['xception', 'efficientnet']
        };
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
