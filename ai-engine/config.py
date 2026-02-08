"""
Configuration settings for the AI Engine
"""
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
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173,http://localhost:8080"
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse ALLOWED_ORIGINS string to list"""
        if not self.ALLOWED_ORIGINS.strip():
            return []
        return [x.strip() for x in self.ALLOWED_ORIGINS.split(",")]
    
    # Model Paths
    MODELS_DIR: str = os.path.join(os.path.dirname(__file__), "models", "weights")
    XCEPTION_MODEL_PATH: str = os.path.join(MODELS_DIR, "xception_deepfake.h5")
    EFFICIENTNET_MODEL_PATH: str = os.path.join(MODELS_DIR, "efficientnet_deepfake.h5")
    LSTM_MODEL_PATH: str = os.path.join(MODELS_DIR, "cnn_lstm_deepfake.h5")
    
    # Upload Settings
    UPLOAD_DIR: str = os.path.join(os.path.dirname(__file__), "uploads")
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".webm", ".mkv"]
    
    # Processing Settings
    IMAGE_SIZE: tuple = (299, 299)  # For Xception
    EFFICIENTNET_SIZE: tuple = (380, 380)  # For EfficientNet-B4
    BATCH_SIZE: int = 16
    VIDEO_FRAME_SAMPLE_RATE: int = 10  # Process every Nth frame
    MAX_VIDEO_FRAMES: int = 100
    
    # Detection Thresholds
    FAKE_THRESHOLD: float = 0.5
    HIGH_CONFIDENCE_THRESHOLD: float = 0.85
    LOW_CONFIDENCE_THRESHOLD: float = 0.35
    
    # Ensemble Weights
    XCEPTION_WEIGHT: float = 0.35
    EFFICIENTNET_WEIGHT: float = 0.40
    LSTM_WEIGHT: float = 0.25
    
    # Forensics Settings
    FACE_DETECTION_CONFIDENCE: float = 0.5
    BLINK_THRESHOLD: float = 0.25
    LIP_SYNC_WINDOW: int = 5
    
    # XAI Settings
    GRADCAM_LAYER: str = "block14_sepconv2_act"  # For Xception
    LIME_NUM_SAMPLES: int = 1000
    LIME_NUM_FEATURES: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Create necessary directories
os.makedirs(settings.MODELS_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
