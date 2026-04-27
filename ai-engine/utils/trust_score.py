"""
Trust Score & Confidence Level Utilities — EDDS AI Engine

Computes trust_score, temporal consistency, and self-aware confidence
level from existing detection outputs. Added as a non-intrusive module
that is called at the end of detection without modifying core logic.
"""
import numpy as np
from typing import List, Optional, Tuple


def compute_agreement_score(fake_probabilities: List[float]) -> float:
    """
    Compute model agreement score (0–1) from individual model fake_probabilities.
    Higher score = models agree. Lower score = models disagree.

    Uses variance-based approach when multiple models exist.
    """
    if len(fake_probabilities) < 2:
        # Single model or no models — default moderate agreement
        return 0.80

    variance = float(np.var(fake_probabilities))
    # Max realistic variance for probabilities in [0,1] is ~0.25
    # Map: variance=0 → agreement=1.0, variance>=0.25 → agreement=0.0
    agreement = max(0.0, 1.0 - (variance / 0.25))
    return round(agreement, 4)


def compute_trust_score(
    confidence: float,
    agreement_score: float,
    forensic_score: Optional[float] = None,
) -> float:
    """
    Composite trust score (0–100) combining:
      - 50% detection confidence
      - 30% model agreement
      - 20% forensic consistency

    Formula:
      trust_score = (0.5 * confidence + 0.3 * agreement + 0.2 * forensic) * 100
    """
    # If no forensic score is available, use a reasonable placeholder
    if forensic_score is None:
        forensic_score = 0.75  # neutral default

    raw = (0.5 * confidence) + (0.3 * agreement_score) + (0.2 * forensic_score)
    return round(min(100.0, max(0.0, raw * 100)), 2)


def compute_temporal_variance(frame_scores: List[float]) -> Tuple[Optional[float], Optional[str]]:
    """
    Compute temporal consistency metrics for video analysis.

    Returns:
        (temporal_variance, temporal_label)
        Both are None for images (no frames).
    """
    if not frame_scores or len(frame_scores) < 2:
        return None, None

    variance = float(np.var(frame_scores))
    variance = round(variance, 4)

    if variance < 0.05:
        label = "Stable"
    elif variance <= 0.15:
        label = "Moderate Variation"
    else:
        label = "High Instability"

    return variance, label


def compute_confidence_level(
    confidence_pct: float,
    temporal_variance: Optional[float] = None,
) -> str:
    """
    Self-aware confidence output.

    Args:
        confidence_pct: Overall confidence as a percentage (0–100).
        temporal_variance: Variance across frames (None for images).

    Returns:
        Human-readable confidence label.
    """
    high_variance = (temporal_variance is not None and temporal_variance > 0.15)

    if confidence_pct < 50 or high_variance:
        return "Low Confidence - Human Review Recommended"
    elif confidence_pct < 75:
        return "Moderate Confidence"
    else:
        # Also downgrade if moderate temporal variance
        if temporal_variance is not None and temporal_variance > 0.05:
            return "Moderate Confidence"
        return "High Confidence"
