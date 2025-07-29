"""
Logging Utility
---------------
Central logging configuration for all modules.

Author: Basel Amr Barakat
Email: baselamr52@gmail.com
"""
import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_logger(name: str):
    """Create and configure a logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(LOG_DIR / f"{name}.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
