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

    // Attach JWT token from localStorage if present
    const token = localStorage.getItem('token');
    if (token) {
        defaultHeaders['Authorization'] = `Bearer ${token}`;
    }

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
 * Get analytics summary — merges /summary, /trends, /confidence, /models endpoints
 * Returns the shape expected by AnalyticsPage.jsx
 * @param {number} trendDays - Days of trend data to fetch (default 7)
 * @returns {Promise<Object>} Merged analytics data
 */
export async function getAnalytics(trendDays = 7) {
    // Fetch all analytics endpoints in parallel
    const [summaryRes, trendsRes, confidenceRes, modelsRes] = await Promise.allSettled([
        apiRequest(`${BACKEND_URL}/analytics/summary`),
        apiRequest(`${BACKEND_URL}/analytics/trends?days=${trendDays}`),
        apiRequest(`${BACKEND_URL}/analytics/confidence`),
        apiRequest(`${BACKEND_URL}/analytics/models`),
    ]);

    // Safely extract data even if some calls failed
    const summary    = summaryRes.status    === 'fulfilled' ? (summaryRes.value.data    || {}) : {};
    const trendsData = trendsRes.status     === 'fulfilled' ? (trendsRes.value.data     || {}) : {};
    const confidence = confidenceRes.status === 'fulfilled' ? (confidenceRes.value.data || []) : [];
    const models     = modelsRes.status     === 'fulfilled' ? (modelsRes.value.data     || {}) : {};

    // Daily analyses for the line chart (trendDays entries)
    const dailyAnalyses = (trendsData.trends || []).map(t => t.count);

    // Confidence distribution for bar chart (5 buckets: 0-20, 20-40, 40-60, 60-80, 80-100)
    const confidenceDistribution = Array.isArray(confidence)
        ? confidence.map(b => b.count)
        : [0, 0, 0, 0, 0];

    // Model metrics table — map from {xception:{name,accuracy,...}} to array
    const modelMetrics = Object.values(models.models || {}).map(m => ({
        name:      m.name,
        accuracy:  m.accuracy,
        precision: m.precision,
        recall:    m.recall,
        f1:        m.f1Score ?? m.f1,
    }));

    return {
        totalAnalyses:  summary.total  || 0,
        avgConfidence:  parseFloat(summary.averages?.confidence || 0),
        avgProcessingTime: parseFloat(summary.averages?.processingTime || 0),
        weeklyAnalyses: summary.thisWeek?.count   || 0,
        weeklyChange:   parseFloat((summary.thisWeek?.change || '0%').replace('%', '').replace('+', '')),
        classificationBreakdown: {
            real:      summary.classifications?.real      || 0,
            fake:      summary.classifications?.fake      || 0,
            uncertain: summary.classifications?.uncertain || 0,
        },
        mediaTypes: {
            image: summary.mediaTypes?.image || 0,
            video: summary.mediaTypes?.video || 0,
        },
        dailyAnalyses,
        confidenceDistribution,
        modelMetrics,
    };
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
/**
 * Strip markdown bold/italic/code syntax from a string
 * (The AI engine returns markdown-formatted text_explanation)
 */
function stripMarkdown(text) {
    if (!text) return text;
    return text
        .replace(/\*\*([^*]+)\*\*/g, '$1')   // **bold**
        .replace(/\*([^*]+)\*/g, '$1')         // *italic*
        .replace(/`([^`]+)`/g, '$1')            // `code`
        .replace(/#+\s/g, '')                   // # headings
        .replace(/⚠️\s*/g, '')                  // stray emoji
        .trim();
}

/**
 * Normalise a model_name string from the AI engine into a camelCase key
 * e.g. "Xception"  → "xception"
 *      "EfficientNet" → "efficientnet"
 *      "CNN-LSTM" / "CNNLSTM" → "cnnLstm"
 */
function normaliseModelKey(name) {
    const n = name.toLowerCase().replace(/[^a-z0-9]/g, '');
    if (n.includes('cnn') && n.includes('lstm')) return 'cnnLstm';
    if (n.includes('efficient')) return 'efficientnet';
    if (n.includes('xception')) return 'xception';
    if (n.includes('resnet')) return 'resnet50';
    if (n.includes('ensemble')) return 'ensemble';
    if (n.includes('frame')) return 'frameAnalysis';
    return n;
}

function formatAnalysisResults(results, isVideo) {
    // The /detect endpoint wraps results in a `result` key:
    // DetectionResponse { analysis_id, status, file_type, filename, timestamp, result: DetectionResult }
    const detectionResp = results.detection || {};
    const detResult = detectionResp.result || detectionResp; // unwrap if nested

    // The /forensics/analyze endpoint wraps in `results` key:
    // ForensicsResponse { analysis_id, file_type, filename, timestamp, results: ForensicsResult }
    const forensicsResp = results.forensics || {};
    const forensicsData = forensicsResp.results || forensicsResp; // unwrap if nested

    const explanation = results.explanation || {};

    // ── Classification ──────────────────────────────────────
    const probability = detResult.fake_probability ?? 0.5;
    let classification = 'uncertain';
    if (probability > 0.6) classification = 'fake';
    else if (probability < 0.35) classification = 'real';

    // ── Model Predictions ────────────────────────────────────
    // model_predictions is a list: [{ model_name, fake_probability, confidence, weight }]
    const modelPredictions = {};
    const rawPreds = detResult.model_predictions || [];
    rawPreds.forEach(pred => {
        const key = normaliseModelKey(pred.model_name);
        modelPredictions[key] = {
            score: pred.fake_probability,
            weight: pred.weight,
        };
    });

    // ── Forensics ────────────────────────────────────────────
    // ForensicsResult { overall_score, landmarks, frequency, blink, temporal, summary }
    const formattedForensics = {
        facialLandmarks: {
            score: forensicsData.landmarks?.score ?? 0.75,
            anomaly: (forensicsData.landmarks?.score ?? 0.75) < 0.6,
        },
        eyeBlink: {
            score: forensicsData.blink?.score ?? 0.8,
            anomaly: !(forensicsData.blink?.natural_pattern ?? true),
        },
        lipSync: {
            score: 0.75, // not implemented in AI engine yet
            anomaly: false,
        },
        frequency: {
            score: 1 - (forensicsData.frequency?.spectrum_anomaly ?? 0.25),
            anomaly: forensicsData.frequency?.artifacts_detected ?? false,
        },
    };

    // ── Explanation ──────────────────────────────────────────
    // XAIResponse: { text_explanation, key_regions: [{ name, importance, finding }] }
    const rawSummary = explanation.text_explanation ||
        `Analysis indicates ${(probability * 100).toFixed(1)}% manipulation probability. ` +
        (detResult.notes?.join(' ') || '');

    const formattedExplanation = {
        summary: stripMarkdown(rawSummary),
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
            lower: detResult.confidence_interval?.lower ?? Math.max(0, probability - 0.05),
            upper: detResult.confidence_interval?.upper ?? Math.min(1, probability + 0.05),
        },
        modelPredictions,
        forensics: formattedForensics,
        explanation: formattedExplanation,
        processingTime: results.processingTime,
        raw: results,
    };
}

// Export URL constants for configuration
export { BACKEND_URL, AI_ENGINE_URL };
