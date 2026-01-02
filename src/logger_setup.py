import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from scripts.validator import ensure_directory_exists


def setup_logger() -> logging.Logger:
    """
    Configures and returns a logger instance.
    Uses validator to ensure logs directory exists.
    """
    try:
        ensure_directory_exists(config.LOGS_DIR)
    except Exception as e:
        raise RuntimeError(f"Failed to ensure logs directory exists: {e}")

    logger = logging.getLogger('pipeline_logger')
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        return logger

    try:
        fh = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
    except Exception as e:
        raise RuntimeError(f"Failed to set up logger handlers: {e}")

    return logger