"""
Enhanced logger for ComplexFuncBench that combines file logging with centralized config.
Extends the project's standardized logger system while preserving benchmark functionality.
"""
import logging
import sys
import os

# Import base logger from project's utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from src.utils.logger import get_logger as get_base_logger


class Logger:
    """
    Enhanced logger that combines file logging with centralized config.
    Maintains backward compatibility with benchmark code while using unified logging.
    """
    def __init__(self, name='cfb_logger', log_file='test.log', level=None):
        """
        Initialize logger with file output capability.
        
        Args:
            name: Logger name
            log_file: Path to log file for persistent logging
            level: Optional log level override (if None, uses config.toml setting)
        """
        # Get base logger with config-driven level management
        self.logger = get_base_logger(name, level)
        
        # Add file handler for benchmark-specific file logging
        if log_file:
            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.logger.level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, msg):
        """Log debug message."""
        self.logger.debug(msg)

    def info(self, msg):
        """Log info message."""
        self.logger.info(msg)

    def warning(self, msg):
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg):
        """Log error message."""
        self.logger.error(msg)

    def critical(self, msg):
        """Log critical message."""
        self.logger.critical(msg)


# Example usage
if __name__ == "__main__":
    log = Logger(name="test_logger", log_file="logs/test.log", level=logging.DEBUG)
    log.debug("This is a debug message")
    log.info("This is an info message")
    log.warning("This is a warning message")
    log.error("This is an error message")
