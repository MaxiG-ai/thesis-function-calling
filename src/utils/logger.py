"""
Custom logging configuration for thesis-function-calling project.

Provides centralized logger creation with consistent formatting and level configuration.
Separates high-level progress (INFO) from detailed technical information (DEBUG).
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get or create a logger with consistent formatting.
    
    Args:
        name: Name for the logger (typically module name)
        level: Optional override for log level. If None, uses INFO as default.
               Can be set globally via config.toml's logging_level setting.
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured (avoid duplicate handlers)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Set level
        if level:
            logger.setLevel(getattr(logging, level.upper()))
        else:
            logger.setLevel(logging.INFO)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger


def set_global_log_level(level: str):
    """
    Update log level for all loggers in the thesis-function-calling namespace.
    
    Args:
        level: Log level as string (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Update all existing loggers
    for name in logging.root.manager.loggerDict:
        if isinstance(logging.root.manager.loggerDict[name], logging.Logger):
            logger = logging.getLogger(name)
            logger.setLevel(log_level)
            for handler in logger.handlers:
                handler.setLevel(log_level)
