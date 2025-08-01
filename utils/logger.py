import logging
import os
import datetime

def get_logger(name, file_path=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if file_path:
        file_handler = logging.FileHandler(os.path.join(file_path, f"{datetime.datetime.now().strftime('%Y-%m-%d')}.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def logger_info(logger, message):
    logger.info(message)

def logger_debug(logger, message):
    logger.debug(message)

def logger_warning(logger, message):
    logger.warning(message)

def logger_error(logger, message): 
    logger.error(message)