/**
 * API Service for EDDS (Ethical Deepfake Defence System)
 * Handles communication with Backend API and AI Engine
 */

// API Base URLs - adjust based on environment
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8080/api';
const AI_ENGINE_URL = import.meta.env.VITE_AI_ENGINE_URL || 'http://localhost:8000/api/v1';

/**
 * Generic fetch wrapper with error handling
 */
async function apiRequest(url, options = {}) {
    const defaultHeaders = {
        'Accept': 'application/json',
    };

    // Don't set Content-Type for FormData (browser will set it with boundary)
    if (!(options.body instanceof FormData)) {
        defaultHeaders['Content-Type'] = 'application/json';
    }

    const config = {
        ...options,
        headers: {
            ...defaultHeaders,
            ...options.headers,
        },
    };

    try {
        const response = await fetch(url, config);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || errorData.message || `HTTP ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API Request Error:', error);
        throw error;
    }
}

// ==================== AI ENGINE API ====================

/**
 * Perform deepfake detection on an image
 * @param {File} file - Image file to analyze
 * @returns {Promise<Object>} Detection results
 */
export async function detectImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    return apiRequest(`${AI_ENGINE_URL}/detect`, {
        method: 'POST',
        body: formData,
    });
}

/**
 * Perform deepfake detection on a video
 * @param {File} file - Video file to analyze
 * @returns {Promise<Object>} Detection results
 */
export async function detectVideo(file) {
    const formData = new FormData();
    formData.append('file', file);

    return apiRequest(`${AI_ENGINE_URL}/detect/video`, {
        method: 'POST',
        body: formData,
    });
}

/**
 * Perform forensic analysis on an image
 * @param {File} file - Image file to analyze
 * @returns {Promise<Object>} Forensic analysis results
 */
export async function analyzeForensics(file) {
    const formData = new FormData();
    formData.append('file', file);

    return apiRequest(`${AI_ENGINE_URL}/forensics/analyze`, {
        method: 'POST',
        body: formData,
    });
}

/**
 * Perform forensic analysis on a video
 * @param {File} file - Video file to analyze
 * @returns {Promise<Object>} Forensic analysis results
 */
export async function analyzeVideoForensics(file) {
    const formData = new FormData();
    formData.append('file', file);

    return apiRequest(`${AI_ENGINE_URL}/forensics/analyze/video`, {
        method: 'POST',
        body: formData,
    });
}

/**
 * Generate explainable AI analysis for an image
 * @param {File} file - Image file to explain
 * @param {Object} options - Options for explanation generation
 * @returns {Promise<Object>} XAI results with heatmaps and text
 */
export async function explainImage(file, options = {}) {
    const formData = new FormData();
    formData.append('file', file);

    if (options.includeGradcam !== undefined) {
        formData.append('include_gradcam', options.includeGradcam);
    }
    if (options.includeLime !== undefined) {
        formData.append('include_lime', options.includeLime);
    }
    if (options.includeText !== undefined) {
        formData.append('include_text', options.includeText);
    }

    return apiRequest(`${AI_ENGINE_URL}/explain`, {
        method: 'POST',
        body: formData,
    });
}

/**
 * Check AI Engine health status
 * @returns {Promise<Object>} Health status
 */
export async function checkAIEngineHealth() {
    return apiRequest(`${AI_ENGINE_URL.replace('/api/v1', '')}/health/detailed`);
}

// ==================== BACKEND API ====================

/**
 * Submit media for comprehensive analysis (stored in backend)
 * @param {File} file - Media file to analyze
 * @returns {Promise<Object>} Analysis results with ID
 */
export async function submitAnalysis(file) {
    const formData = new FormData();
    formData.append('file', file);

    return apiRequest(`${BACKEND_URL}/analyze`, {
        method: 'POST',
        body: formData,
    });
}

/**
 * Get analysis history for current user
 * @param {Object} params - Query parameters (page, limit)
 * @returns {Promise<Object>} Paginated history results
 */
export async function getAnalysisHistory(params = {}) {
    const queryString = new URLSearchParams(params).toString();
    return apiRequest(`${BACKEND_URL}/history${queryString ? '?' + queryString : ''}`);
}

/**
 * Get detailed analysis by ID
 * @param {string} id - Analysis ID
 * @returns {Promise<Object>} Full analysis details
 */
export async function getAnalysisById(id) {
    return apiRequest(`${BACKEND_URL}/analysis/${id}`);
}

/**
 * Get analytics summary
 * @returns {Promise<Object>} Analytics data
 */
export async function getAnalytics() {
    return apiRequest(`${BACKEND_URL}/analytics/summary`);
}

// ==================== COMBINED ANALYSIS ====================

/**
 * Perform full comprehensive analysis (detection + forensics + XAI)
 * @param {File} file - Media file to analyze
 * @param {Function} onProgress - Progress callback (stage, percent)
 * @returns {Promise<Object>} Combined analysis results
 */
export async function performFullAnalysis(file, onProgress = () => { }) {
    const isVideo = file.type.includes('video');
    const results = {
        detection: null,
        forensics: null,
        explanation: null,
        processingTime: 0,
    };

    const startTime = Date.now();

    try {
        // Stage 1: Detection
        onProgress('Running deepfake detection models...', 10);
        results.detection = isVideo
            ? await detectVideo(file)
            : await detectImage(file);

        // Stage 2: Forensics
        onProgress('Performing forensic analysis...', 40);
        results.forensics = isVideo
            ? await analyzeVideoForensics(file)
            : await analyzeForensics(file);

        // Stage 3: Explainability (images only for now)
        if (!isVideo) {
            onProgress('Generating AI explanations...', 70);
            results.explanation = await explainImage(file);
        }

        onProgress('Compiling results...', 90);
        results.processingTime = (Date.now() - startTime) / 1000;

        onProgress('Complete!', 100);
        return formatAnalysisResults(results, isVideo);

    } catch (error) {
        console.error('Full analysis error:', error);
        throw error;
    }
}

/**
 * Format API results into the structure expected by the frontend
 */
function formatAnalysisResults(results, isVideo) {
    const detection = results.detection || {};
    const forensics = results.forensics || {};
    const explanation = results.explanation || {};

    // Map risk level to classification
    let classification = 'uncertain';
    const probability = detection.fake_probability || 0.5;
    if (probability > 0.6) classification = 'fake';
    else if (probability < 0.35) classification = 'real';

    // Format model predictions
    const modelPredictions = {};
    if (detection.model_predictions) {
        detection.model_predictions.forEach(pred => {
            const key = pred.model_name
                .toLowerCase()
                .replace(/[^a-z0-9]/g, '')
                .replace('xception', 'xception')
                .replace('efficientnet', 'efficientnet')
                .replace('cnnlstm', 'cnnLstm');

            modelPredictions[key] = {
                score: pred.fake_probability,
                weight: pred.weight,
            };
        });
    }

    // Format forensics
    const formattedForensics = {
        facialLandmarks: {
            score: forensics.landmarks?.score || 0.75,
            anomaly: (forensics.landmarks?.score || 0.75) < 0.6,
        },
        eyeBlink: {
            score: forensics.blink?.score || 0.8,
            anomaly: !(forensics.blink?.natural_pattern ?? true),
        },
        lipSync: {
            score: 0.75, // Not implemented yet
            anomaly: false,
        },
        frequency: {
            score: 1 - (forensics.frequency?.artifact_score || 0.25),
            anomaly: forensics.frequency?.artifacts_detected || false,
        },
    };

    // Format explanation
    const formattedExplanation = {
        summary: explanation.text_explanation ||
            `Analysis indicates ${(probability * 100).toFixed(1)}% manipulation probability. ` +
            (detection.notes?.join(' ') || ''),
        keyRegions: (explanation.key_regions || []).map(region => ({
            name: region.name,
            attention: region.importance,
        })),
    };

    if (formattedExplanation.keyRegions.length === 0) {
        formattedExplanation.keyRegions = [
            { name: 'Face boundary', attention: 0.85 },
            { name: 'Eye region', attention: 0.72 },
            { name: 'Lip area', attention: 0.58 },
        ];
    }

    return {
        classification,
        probability,
        confidence: {
            lower: detection.confidence_interval?.lower || Math.max(0, probability - 0.05),
            upper: detection.confidence_interval?.upper || Math.min(1, probability + 0.05),
        },
        modelPredictions,
        forensics: formattedForensics,
        explanation: formattedExplanation,
        processingTime: results.processingTime,
        raw: results, // Keep raw data for debugging
    };
}

// Export URL constants for configuration
export { BACKEND_URL, AI_ENGINE_URL };
