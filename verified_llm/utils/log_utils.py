import logging
import os
from datetime import datetime

LOG_NAME = "decen_train"
_logger_instance = None

def setup_logger(log_dir="logs", level=logging.INFO) -> logging.Logger:
    """
    Setup logger with singleton pattern to avoid creating multiple log files
    """
    global _logger_instance
    
    # Return existing logger if already created
    if _logger_instance is not None:
        return _logger_instance
    
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{LOG_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger(LOG_NAME)
    logger.setLevel(level)
    logger.propagate = False

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Store the instance
    _logger_instance = logger
    
    # Log the initialization
    logger.info(f"Logger initialized. Log file: {log_file}")
    
    return logger

def get_logger() -> logging.Logger:
    """
    Get the global logger instance, creating it if necessary
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = setup_logger()
    return _logger_instance

# Global logger instance - use get_logger() instead of direct setup
g_logger = get_logger()

if __name__ == '__main__':
    # Test the logger
    logger = get_logger()
    logger.warning("This is a test warning message")
    logger.info("This is a test info message")
    logger.debug("This is a test debug message")
    print("Logger test completed. Check the logs directory for the log file.")