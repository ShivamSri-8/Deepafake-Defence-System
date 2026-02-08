"""
EDDS AI Engine - Main FastAPI Application
Ethical Deepfake Defence System
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

from config import settings
from routers import detection, forensics, xai, health
from utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"📂 Models directory: {settings.MODELS_DIR}")
    logger.info(f"📤 Uploads directory: {settings.UPLOAD_DIR}")
    
    # Check if models exist
    models_exist = all([
        os.path.exists(settings.XCEPTION_MODEL_PATH),
        os.path.exists(settings.EFFICIENTNET_MODEL_PATH),
        os.path.exists(settings.LSTM_MODEL_PATH)
    ])
    
    if not models_exist:
        logger.warning("⚠️ Pre-trained models not found. Running in simulation mode.")
        logger.info("💡 To enable real detection, place model weights in: " + settings.MODELS_DIR)
    else:
        logger.info("✅ All pre-trained models loaded successfully")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down AI Engine")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="""
    ## Ethical Deepfake Defence System - AI Engine
    
    This API provides deepfake detection, forensic analysis, and explainable AI capabilities.
    
    ### Features:
    - **Detection**: Multi-model ensemble for image/video deepfake detection
    - **Forensics**: Facial landmark analysis, blink detection, frequency analysis
    - **XAI**: Grad-CAM heatmaps, LIME explanations, human-readable insights
    
    ### Important Notice:
    Results are probabilistic assessments and should NOT be used as sole evidence.
    """,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for uploads
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(detection.router, prefix="/api/v1", tags=["Detection"])
app.include_router(forensics.router, prefix="/api/v1", tags=["Forensics"])
app.include_router(xai.router, prefix="/api/v1", tags=["Explainability"])


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "documentation": "/docs",
        "health_check": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
