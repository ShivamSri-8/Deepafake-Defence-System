"""
Configuration settings for the AI Engine
"""
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # API Settings
    APP_NAME: str = "EDDS AI Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS Settings
    ALLOWED_ORIGINS: str = "*"

    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse ALLOWED_ORIGINS string to list"""
        if not self.ALLOWED_ORIGINS.strip():
            return []
        return [x.strip() for x in self.ALLOWED_ORIGINS.split(",")]

    # ── Model Paths ──────────────────────────────────────────────────────────
    MODELS_DIR: str = os.path.join(os.path.dirname(__file__), "models", "weights")
    MODEL_PATH: str = os.path.join(MODELS_DIR, "efficientnet_deepfake.pt")

    # ── Inference Backend ─────────────────────────────────────────────────────
    # Enforcing PyTorch mode for production. TensorFlow is removed.
    USE_PYTORCH: bool = True

    # ── Upload Settings ───────────────────────────────────────────────────────
    UPLOAD_DIR: str = os.path.join(os.path.dirname(__file__), "uploads")
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100 MB
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".webm", ".mkv"]

    # ── Processing Settings ───────────────────────────────────────────────────
    IMAGE_SIZE: List[int] = [299, 299]        # Xception / ResNet50
    EFFICIENTNET_SIZE: List[int] = [380, 380] # EfficientNet-B4
    BATCH_SIZE: int = 16
    VIDEO_FRAME_SAMPLE_RATE: int = 10     # Process every Nth frame
    MAX_VIDEO_FRAMES: int = 100

    # ── Classification Thresholds (per improvement plan) ─────────────────────
    #   <= 0.40  → Authentic
    #   0.40–0.60 → Suspicious
    #   >= 0.60  → Deepfake
    FAKE_THRESHOLD: float = 0.60
    SUSPICIOUS_THRESHOLD: float = 0.40
    HIGH_CONFIDENCE_THRESHOLD: float = 0.85
    LOW_CONFIDENCE_THRESHOLD: float = 0.40

    # ── Classification Thresholds (per improvement plan) ─────────────────────

    # ── Forensics Settings ────────────────────────────────────────────────────
    FACE_DETECTION_CONFIDENCE: float = 0.5
    BLINK_THRESHOLD: float = 0.25
    LIP_SYNC_WINDOW: int = 5

    # ── XAI Settings ──────────────────────────────────────────────────────────
    GRADCAM_LAYER: str = "features.7"  # PyTorch EfficientNet target layer
    LIME_NUM_SAMPLES: int = 100
    LIME_NUM_FEATURES: int = 10

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = 'ignore'   # silently ignore unknown .env keys (e.g. old LSTM_WEIGHT)


settings = Settings()

# Create necessary directories
os.makedirs(settings.MODELS_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
