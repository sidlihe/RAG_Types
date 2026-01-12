import logging
import os
from datetime import datetime

def get_logger(name: str):
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Generate log filename with timestamp
    log_filename = datetime.now().strftime("%Y-%m-%d") + ".log"
    log_filepath = os.path.join(log_dir, log_filename)

    # Configure logging
    logger = logging.getLogger(name)
    
    # Prevent adding multiple handlers if logger is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # File Handler
        file_handler = logging.FileHandler(log_filepath)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger