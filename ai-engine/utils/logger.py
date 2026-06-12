"""
Logger utility with colored output
"""
import sys
from loguru import logger


def _safe_reconfigure_streams():
    """Reconfigure sys.stdout/stderr to handle UTF-8 without closing the original streams."""
    try:
        # Python 3.7+ supports reconfigure() which is non-destructive
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def setup_logger(name: str = None):
    """Configure and return a logger instance"""

    # Safely reconfigure streams for UTF-8 support (non-destructive)
    _safe_reconfigure_streams()

    # Remove default handler
    logger.remove()

    # Add custom handler with formatting — write to stderr so we don't
    # interfere with uvicorn's stdout-based logging formatter.
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG",
        colorize=True,
        enqueue=False,
    )
    
    # Add file handler for errors
    logger.add(
        "logs/error.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
    )
    
    # Add file handler for all logs
    logger.add(
        "logs/app.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
    )
    
    return logger
