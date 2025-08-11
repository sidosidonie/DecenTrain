import logging
import os
from datetime import datetime
import atexit

LOG_NAME = "decen_train"
_logger_instance = None

class FixedFileHandler(logging.FileHandler):
    """
    A file handler that uses a fixed filename and rotates logs
    """
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        super().__init__(filename, mode, encoding, delay)

def setup_logger(log_dir="logs", level=logging.INFO, use_timestamp=False) -> logging.Logger:
    """
    Setup logger with better file management
    
    Args:
        log_dir (str): Directory for log files
        level: Logging level
        use_timestamp (bool): If True, use timestamp in filename, otherwise use fixed name
    """
    global _logger_instance
    
    # Return existing logger if already created
    if _logger_instance is not None:
        return _logger_instance
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Use fixed filename to avoid creating multiple empty files
    if use_timestamp:
        log_file = os.path.join(log_dir, f"{LOG_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    else:
        log_file = os.path.join(log_dir, f"{LOG_NAME}.log")

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

    # Add file handler with fixed filename
    try:
        file_handler = FixedFileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create file handler: {e}")

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

def close_logger():
    """
    Properly close the logger and its handlers
    """
    global _logger_instance
    if _logger_instance is not None:
        for handler in _logger_instance.handlers[:]:
            handler.close()
            _logger_instance.removeHandler(handler)
        _logger_instance = None

# Register cleanup function to be called on exit
atexit.register(close_logger)

# Global logger instance
g_logger = get_logger()

if __name__ == '__main__':
    # Test the logger
    logger = get_logger()
    logger.warning("This is a test warning message")
    logger.info("This is a test info message")
    logger.debug("This is a test debug message")
    print("Logger test completed. Check the logs directory for the log file.")
