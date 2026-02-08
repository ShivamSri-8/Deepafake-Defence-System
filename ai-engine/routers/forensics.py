"""
Forensics router - Forensic analysis endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
import uuid
from datetime import datetime

from models.schemas import ForensicsResponse
from services.forensics import ForensicsAnalyzer
from utils.file_handler import save_upload_file, get_file_type
from utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

# Initialize forensics analyzer
forensics = ForensicsAnalyzer()


@router.post("/forensics/analyze", response_model=ForensicsResponse)
async def analyze_forensics(
    file: UploadFile = File(...),
    include_landmarks: bool = True,
    include_blink: bool = True,
    include_frequency: bool = True,
    include_temporal: bool = True
):
    """
    Perform forensic analysis on an image or video.
    
    Analysis includes:
    - **Facial landmark consistency**: Detects unnatural facial geometry
    - **Blink detection**: Analyzes eye blink patterns (video only)
    - **Frequency analysis**: Detects GAN artifacts in frequency domain
    - **Temporal consistency**: Analyzes frame-to-frame coherence (video only)
    """
    logger.info(f"🔬 Starting forensic analysis: {file.filename}")
    
    analysis_id = str(uuid.uuid4())
    
    try:
        # Save file
        file_path, file_id = await save_upload_file(file, subfolder="forensics")
        file_type = get_file_type(file.filename)
        
        # Perform forensic analysis
        if file_type == "image":
            result = await forensics.analyze_image(
                file_path,
                include_landmarks=include_landmarks,
                include_frequency=include_frequency
            )
        elif file_type == "video":
            result = await forensics.analyze_video(
                file_path,
                include_landmarks=include_landmarks,
                include_blink=include_blink,
                include_frequency=include_frequency,
                include_temporal=include_temporal
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        response = ForensicsResponse(
            analysis_id=analysis_id,
            file_type=file_type,
            filename=file.filename,
            timestamp=datetime.utcnow().isoformat(),
            results=result
        )
        
        logger.info(f"✅ Forensic analysis complete: {analysis_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Forensics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forensics/landmarks")
async def analyze_landmarks(file: UploadFile = File(...)):
    """Analyze facial landmarks for inconsistencies"""
    file_path, _ = await save_upload_file(file, subfolder="forensics")
    result = await forensics.analyze_landmarks(file_path)
    return result


@router.post("/forensics/frequency")
async def analyze_frequency(file: UploadFile = File(...)):
    """Analyze frequency domain for GAN artifacts"""
    file_path, _ = await save_upload_file(file, subfolder="forensics")
    result = await forensics.analyze_frequency(file_path)
    return result


@router.post("/forensics/blink")
async def analyze_blink(file: UploadFile = File(...)):
    """Analyze eye blink patterns in video"""
    file_type = get_file_type(file.filename)
    if file_type != "video":
        raise HTTPException(status_code=400, detail="Blink analysis requires video input")
    
    file_path, _ = await save_upload_file(file, subfolder="forensics")
    result = await forensics.analyze_blink_patterns(file_path)
    return result
