# src/logger_setup.py
import logging
import os
import sys
from datetime import datetime
from src.validator import ensure_directory_exists

def setup_logger(config_log_dir=None, log_name='pipeline_logger') -> logging.Logger:
    """
    Configures and returns a logger instance.
    
    Args:
        config_log_dir (str, optional): Custom path for logs. Defaults to 'logs' in project root.
        log_name (str): Name of the logger.
    """
    if config_log_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(project_root, "logs")
    else:
        log_dir = config_log_dir

    log_filename = f"{datetime.now().strftime('%Y_%m_%d')}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    try:
        ensure_directory_exists(log_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to ensure logs directory exists: {e}")

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        return logger

    try:
        fh = logging.FileHandler(log_filepath, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Format
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
    
    except Exception as e:
        raise RuntimeError(f"Failed to set up logger handlers: {e}")

    return logger