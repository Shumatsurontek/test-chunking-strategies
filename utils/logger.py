import logging
import os
import datetime

logger = logging.getLogger("chunking")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def init_logger(file_path=None):
    if file_path:
        os.makedirs(file_path, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(file_path, f"{datetime.datetime.now().strftime('%Y-%m-%d')}.log")
        )
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            logger.addHandler(file_handler)

def log_info(message):
    logger.info(message)

def log_debug(message):
    logger.debug(message)

def log_warning(message):
    logger.warning(message)

def log_error(message):
    logger.error(message)