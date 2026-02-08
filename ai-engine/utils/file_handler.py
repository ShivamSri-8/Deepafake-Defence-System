"""
File handling utilities
"""
import os
import uuid
import aiofiles
from fastapi import UploadFile, HTTPException
from typing import Tuple
from config import settings


async def save_upload_file(file: UploadFile, subfolder: str = "") -> Tuple[str, str]:
    """
    Save an uploaded file and return the file path and unique ID
    """
    # Generate unique ID for the file
    file_id = str(uuid.uuid4())
    
    # Get file extension
    _, ext = os.path.splitext(file.filename)
    ext = ext.lower()
    
    # Validate file extension
    allowed_extensions = settings.ALLOWED_IMAGE_EXTENSIONS + settings.ALLOWED_VIDEO_EXTENSIONS
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Allowed types: {allowed_extensions}"
        )
    
    # Create upload directory if needed
    upload_dir = os.path.join(settings.UPLOAD_DIR, subfolder) if subfolder else settings.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    
    # Create file path
    filename = f"{file_id}{ext}"
    file_path = os.path.join(upload_dir, filename)
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.0f}MB"
        )
    
    # Save file
    async with aiofiles.open(file_path, "wb") as buffer:
        content = await file.read()
        await buffer.write(content)
    
    return file_path, file_id


def get_file_type(filename: str) -> str:
    """Determine if file is image or video"""
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    
    if ext in settings.ALLOWED_IMAGE_EXTENSIONS:
        return "image"
    elif ext in settings.ALLOWED_VIDEO_EXTENSIONS:
        return "video"
    else:
        return "unknown"


def cleanup_file(file_path: str) -> bool:
    """Delete a file if it exists"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception:
        return False
