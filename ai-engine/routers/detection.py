"""
Detection router - Main deepfake detection endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import Optional
import uuid
from datetime import datetime

from models.schemas import (
    DetectionRequest,
    DetectionResponse,
    DetectionResult,
    ModelPrediction,
    ConfidenceInterval
)
from services.detector import DeepfakeDetector
from utils.file_handler import save_upload_file, get_file_type, cleanup_file
from utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

# Initialize detector
detector = DeepfakeDetector()


@router.post("/detect", response_model=DetectionResponse)
async def detect_deepfake(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Analyze an image or video for deepfake manipulation.
    
    - **file**: Image (jpg, png, webp) or Video (mp4, avi, mov) file
    
    Returns probabilistic detection results with confidence intervals.
    """
    logger.info(f"📥 Received file: {file.filename}")
    
    # Generate analysis ID
    analysis_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file
        file_path, file_id = await save_upload_file(file, subfolder="detection")
        file_type = get_file_type(file.filename)
        
        logger.info(f"💾 Saved file: {file_path} (type: {file_type})")
        
        # Perform detection
        if file_type == "image":
            result = await detector.detect_image(file_path)
        elif file_type == "video":
            result = await detector.detect_video(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Schedule file cleanup (optional - keep for history)
        # if background_tasks:
        #     background_tasks.add_task(cleanup_file, file_path)
        
        # Build response
        response = DetectionResponse(
            analysis_id=analysis_id,
            status="completed",
            file_type=file_type,
            filename=file.filename,
            timestamp=datetime.utcnow().isoformat(),
            result=result,
            disclaimer="This is a probabilistic assessment. Results should not be used as sole evidence."
        )
        
        logger.info(f"✅ Analysis complete: {analysis_id} - Fake probability: {result.fake_probability:.2%}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/batch")
async def detect_batch(
    files: list[UploadFile] = File(...)
):
    """
    Analyze multiple files for deepfake manipulation.
    Limited to 10 files per batch.
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    
    for file in files:
        try:
            # Save and analyze each file
            file_path, file_id = await save_upload_file(file, subfolder="batch")
            file_type = get_file_type(file.filename)
            
            if file_type == "image":
                result = await detector.detect_image(file_path)
            elif file_type == "video":
                result = await detector.detect_video(file_path)
            else:
                result = None
            
            results.append({
                "filename": file.filename,
                "file_type": file_type,
                "result": result,
                "status": "completed" if result else "unsupported"
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "batch_id": str(uuid.uuid4()),
        "total_files": len(files),
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/detect/{analysis_id}")
async def get_detection_result(analysis_id: str):
    """
    Retrieve a previous detection result by analysis ID.
    (Placeholder - would need database integration)
    """
    # This would normally fetch from database
    return {
        "analysis_id": analysis_id,
        "status": "not_found",
        "message": "Result retrieval requires database integration"
    }
