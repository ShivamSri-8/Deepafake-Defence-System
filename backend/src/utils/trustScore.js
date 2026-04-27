/**
 * Trust Score Utilities — EDDS Backend
 *
 * Mirrors the Python ai-engine/utils/trust_score.py logic.
 * Used to enrich the backend's simulation-mode results with
 * trust_score, temporal_variance, temporal_label, and confidence_level.
 */

/**
 * Compute model agreement score (0–1).
 * Higher = models agree more.
 */
function computeAgreementScore(fakeProbabilities) {
    if (!fakeProbabilities || fakeProbabilities.length < 2) {
        return 0.80; // single-model default
    }
    const mean = fakeProbabilities.reduce((a, b) => a + b, 0) / fakeProbabilities.length;
    const variance = fakeProbabilities.reduce((sum, p) => sum + (p - mean) ** 2, 0) / fakeProbabilities.length;
    return Math.max(0, Math.round((1 - variance / 0.25) * 10000) / 10000);
}

/**
 * Composite trust score (0–100).
 * Formula: (0.5 * confidence + 0.3 * agreement + 0.2 * forensic) * 100
 */
function computeTrustScore(confidence, agreementScore, forensicScore = 0.75) {
    const raw = (0.5 * confidence) + (0.3 * agreementScore) + (0.2 * forensicScore);
    return Math.round(Math.min(100, Math.max(0, raw * 100)) * 100) / 100;
}

/**
 * Temporal variance & label (video only).
 * Returns { variance, label } or { variance: null, label: null } for images.
 */
function computeTemporalVariance(frameScores) {
    if (!frameScores || frameScores.length < 2) {
        return { variance: null, label: null };
    }
    const mean = frameScores.reduce((a, b) => a + b, 0) / frameScores.length;
    const variance = Math.round(
        (frameScores.reduce((sum, s) => sum + (s - mean) ** 2, 0) / frameScores.length) * 10000
    ) / 10000;

    let label;
    if (variance < 0.05) label = 'Stable';
    else if (variance <= 0.15) label = 'Moderate Variation';
    else label = 'High Instability';

    return { variance, label };
}

/**
 * Self-aware confidence label.
 */
function computeConfidenceLevel(confidencePct, temporalVariance = null) {
    const highVariance = temporalVariance !== null && temporalVariance > 0.15;

    if (confidencePct < 50 || highVariance) {
        return 'Low Confidence - Human Review Recommended';
    }
    if (confidencePct < 75) {
        return 'Moderate Confidence';
    }
    if (temporalVariance !== null && temporalVariance > 0.05) {
        return 'Moderate Confidence';
    }
    return 'High Confidence';
}

module.exports = {
    computeAgreementScore,
    computeTrustScore,
    computeTemporalVariance,
    computeConfidenceLevel,
};
