"""
Health check router
"""
from fastapi import APIRouter
from datetime import datetime
import platform
import psutil
import os

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "EDDS AI Engine"
    }


@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system info"""
    # Get memory info
    memory = psutil.virtual_memory()
    
    # Check GPU availability
    gpu_available = False
    gpu_info = None
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            }
    except:
        pass

    # Dynamically reflect actual model load state
    simulation_mode = True
    models_message = "Running with simulated detection results"
    try:
        from routers.detection import detector  # import the shared singleton
        simulation_mode = not detector.models_loaded
        if not simulation_mode:
            models_message = "Real ML models loaded and active"
    except Exception:
        pass
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "EDDS AI Engine",
        "system": {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "memory": {
                "total": f"{memory.total / 1e9:.2f} GB",
                "available": f"{memory.available / 1e9:.2f} GB",
                "used_percent": f"{memory.percent}%"
            }
        },
        "gpu": {
            "available": gpu_available,
            "info": gpu_info
        },
        "models": {
            "simulation_mode": simulation_mode,
            "message": models_message
        }
    }
