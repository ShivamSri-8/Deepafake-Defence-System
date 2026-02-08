"""
Pydantic schemas for API request/response models
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class DetectionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# === Confidence Interval ===
class ConfidenceInterval(BaseModel):
    lower: float = Field(..., ge=0, le=1, description="Lower bound of confidence interval")
    upper: float = Field(..., ge=0, le=1, description="Upper bound of confidence interval")
    confidence_level: float = Field(default=0.95, description="Confidence level (e.g., 0.95 for 95%)")


# === Model Predictions ===
class ModelPrediction(BaseModel):
    model_name: str = Field(..., description="Name of the model")
    fake_probability: float = Field(..., ge=0, le=1, description="Probability of being fake")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    weight: float = Field(..., ge=0, le=1, description="Weight in ensemble")


# === Detection Results ===
class DetectionResult(BaseModel):
    is_fake: bool = Field(..., description="Binary classification result")
    fake_probability: float = Field(..., ge=0, le=1, description="Probability of being fake (0-1)")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence score")
    confidence_interval: ConfidenceInterval = Field(..., description="95% confidence interval")
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")
    model_predictions: List[ModelPrediction] = Field(default=[], description="Individual model predictions")
    face_detected: bool = Field(default=True, description="Whether a face was detected")
    notes: List[str] = Field(default=[], description="Additional notes or warnings")


class DetectionRequest(BaseModel):
    """Request model for detection (used for URL-based detection)"""
    url: Optional[str] = Field(None, description="URL of media to analyze")
    options: Optional[Dict[str, Any]] = Field(default={}, description="Detection options")


class DetectionResponse(BaseModel):
    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: DetectionStatus = Field(..., description="Analysis status")
    file_type: MediaType = Field(..., description="Type of analyzed media")
    filename: str = Field(..., description="Original filename")
    timestamp: str = Field(..., description="Analysis timestamp (ISO format)")
    result: DetectionResult = Field(..., description="Detection results")
    disclaimer: str = Field(
        default="This is a probabilistic assessment. Results should not be used as sole evidence.",
        description="Important disclaimer"
    )


# === Forensics Schemas ===
class LandmarkAnalysis(BaseModel):
    score: float = Field(..., ge=0, le=1, description="Landmark consistency score")
    anomalies: List[str] = Field(default=[], description="Detected anomalies")
    regions: Dict[str, float] = Field(default={}, description="Per-region scores")


class FrequencyAnalysis(BaseModel):
    score: float = Field(..., ge=0, le=1, description="Frequency analysis score")
    artifacts_detected: bool = Field(..., description="Whether GAN artifacts were detected")
    spectrum_anomaly: float = Field(..., ge=0, le=1, description="Spectrum anomaly score")
    description: str = Field(default="", description="Analysis description")


class BlinkAnalysis(BaseModel):
    blink_rate: float = Field(..., description="Blinks per minute")
    natural_pattern: bool = Field(..., description="Whether pattern appears natural")
    score: float = Field(..., ge=0, le=1, description="Blink naturalness score")
    total_blinks: int = Field(..., description="Total blinks detected")
    video_duration: float = Field(..., description="Video duration in seconds")


class TemporalAnalysis(BaseModel):
    consistency_score: float = Field(..., ge=0, le=1, description="Frame-to-frame consistency")
    jitter_detected: bool = Field(..., description="Whether temporal jitter was detected")
    anomalous_frames: List[int] = Field(default=[], description="Indices of anomalous frames")


class ForensicsResult(BaseModel):
    overall_score: float = Field(..., ge=0, le=1, description="Overall forensics score")
    landmarks: Optional[LandmarkAnalysis] = None
    frequency: Optional[FrequencyAnalysis] = None
    blink: Optional[BlinkAnalysis] = None
    temporal: Optional[TemporalAnalysis] = None
    summary: str = Field(..., description="Human-readable summary")


class ForensicsResponse(BaseModel):
    analysis_id: str
    file_type: MediaType
    filename: str
    timestamp: str
    results: ForensicsResult


# === XAI Schemas ===
class GradCAMResult(BaseModel):
    heatmap_url: str = Field(..., description="URL to heatmap image")
    overlay_url: str = Field(..., description="URL to overlay image")
    focus_regions: List[str] = Field(default=[], description="Key focus regions")
    max_activation: float = Field(..., ge=0, le=1, description="Maximum activation value")


class LIMEResult(BaseModel):
    explanation_url: str = Field(..., description="URL to LIME visualization")
    top_features: List[Dict[str, Any]] = Field(default=[], description="Top contributing features")
    positive_regions: int = Field(..., description="Regions contributing to 'fake' classification")
    negative_regions: int = Field(..., description="Regions contributing to 'real' classification")


class KeyRegion(BaseModel):
    name: str = Field(..., description="Region name (e.g., 'left_eye', 'mouth')")
    importance: float = Field(..., ge=0, le=1, description="Importance score")
    finding: str = Field(..., description="What was found in this region")


class XAIResponse(BaseModel):
    explanation_id: str
    filename: str
    timestamp: str
    gradcam: Optional[GradCAMResult] = None
    lime: Optional[LIMEResult] = None
    text_explanation: Optional[str] = None
    key_regions: List[KeyRegion] = Field(default=[])


# === Error Schemas ===
class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional details")
