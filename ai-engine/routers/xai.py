"""
XAI (Explainable AI) router - Explanation generation endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
import uuid
import os
from datetime import datetime

from models.schemas import XAIResponse
from services.explainer import ExplainabilityEngine
from utils.file_handler import save_upload_file, get_file_type
from utils.logger import setup_logger
from config import settings

router = APIRouter()
logger = setup_logger(__name__)

# Initialize explainability engine
explainer = ExplainabilityEngine()


@router.post("/explain", response_model=XAIResponse)
async def generate_explanation(
    file: UploadFile = File(...),
    include_gradcam: bool = True,
    include_lime: bool = True,
    include_text: bool = True
):
    """
    Generate explainable AI outputs for a detection result.
    
    Explanation methods:
    - **Grad-CAM**: Attention heatmap showing model focus areas
    - **LIME**: Superpixel importance visualization
    - **Text**: Human-readable explanation of findings
    """
    logger.info(f"🧠 Generating explanations for: {file.filename}")
    
    explanation_id = str(uuid.uuid4())
    
    try:
        # Save file
        file_path, file_id = await save_upload_file(file, subfolder="xai")
        file_type = get_file_type(file.filename)
        
        if file_type != "image":
            raise HTTPException(
                status_code=400, 
                detail="XAI currently supports images only. For video, submit key frames."
            )
        
        # Generate explanations
        result = await explainer.explain(
            file_path,
            include_gradcam=include_gradcam,
            include_lime=include_lime,
            include_text=include_text
        )
        
        response = XAIResponse(
            explanation_id=explanation_id,
            filename=file.filename,
            timestamp=datetime.utcnow().isoformat(),
            gradcam=result.get("gradcam"),
            lime=result.get("lime"),
            text_explanation=result.get("text_explanation"),
            key_regions=result.get("key_regions", [])
        )
        
        logger.info(f"✅ Explanation generated: {explanation_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ XAI error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain/gradcam")
async def generate_gradcam(file: UploadFile = File(...)):
    """Generate only Grad-CAM heatmap visualization"""
    file_path, _ = await save_upload_file(file, subfolder="xai")
    result = await explainer.generate_gradcam(file_path)
    return result


@router.post("/explain/lime")
async def generate_lime(file: UploadFile = File(...)):
    """Generate only LIME superpixel explanation"""
    file_path, _ = await save_upload_file(file, subfolder="xai")
    result = await explainer.generate_lime(file_path)
    return result


@router.post("/explain/text")
async def generate_text_explanation(file: UploadFile = File(...)):
    """Generate only human-readable text explanation"""
    file_path, _ = await save_upload_file(file, subfolder="xai")
    result = await explainer.generate_text_explanation(file_path)
    return {"explanation": result}


@router.get("/explain/heatmap/{heatmap_id}")
async def get_heatmap_image(heatmap_id: str):
    """Retrieve a generated heatmap image by ID"""
    heatmap_path = os.path.join(settings.UPLOAD_DIR, "xai", f"{heatmap_id}_heatmap.png")
    
    if not os.path.exists(heatmap_path):
        raise HTTPException(status_code=404, detail="Heatmap not found")
    
    return FileResponse(heatmap_path, media_type="image/png")
