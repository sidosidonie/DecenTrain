import logging
import os
from datetime import datetime

LOG_NAME="decen_train"

def setup_logger(log_dir="logs", level=logging.INFO) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{LOG_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger(LOG_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        # ? Add filename and line number here
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Global logger instance
g_logger = setup_logger()

if __name__ == '__main__':
    g_logger.warning("This is a test message")
    g_logger.info("This is a test message")
    g_logger.debug("This is a test message")