import logging
import os
import sys
import codecs
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger(name='discochat'):
    """Set up and configure the logger for the application."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Generate a unique filename for this run
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/discochat_{current_time}.log'

    # Create file handler which logs even debug messages
    file_handler = RotatingFileHandler(log_filename, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler(codecs.getwriter('utf-8')(sys.stdout.buffer))
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log the start of the script
    logger.info(f"Script started. Logging to {log_filename}")

    return logger

# Create the default logger instance
logger = setup_logger()
