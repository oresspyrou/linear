import os
from typing import List
import pandas as pd

def validate_input_file(filepath: str) -> bool:
    """
    Validates that a file exists and is readable.
    
    Args:
        filepath (str): Path to the file.
        
    Raises:
        FileNotFoundError: If file does not exist.
        PermissionError: If file is not readable.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file does not exist at path: {filepath}")
    
    if not os.access(filepath, os.R_OK):
        raise PermissionError(f"The file at {filepath} is not readable (check permissions).")
    
    return True


def validate_csv_columns(filepath: str, required_columns: List[str]) -> bool:
    """
    Checks if a CSV file contains specific columns without loading the whole file.
    
    Args:
        filepath (str): Path to CSV.
        required_columns (List[str]): List of column names that must exist.
    """
    try:
        df: pd.DataFrame = pd.read_csv(filepath, nrows=0)
        missing: List[str] = [col for col in required_columns if col not in df.columns]

        if missing:
            raise ValueError(f"The columns: {missing} are missing from {os.path.basename(filepath)}")

    except pd.errors.EmptyDataError:
        raise ValueError(f"The file {os.path.basename(filepath)} is empty.")
    except pd.errors.ParserError as e:
        raise RuntimeError(f"Failed to parse CSV {os.path.basename(filepath)}. Is it a valid CSV? Error: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV for validation: {e}")
    
    return True


def ensure_directory_exists(dirpath: str) -> None:
    """
    Creates a directory if it does not exist.
    Thread-safe implementation.
    """
    try:
        os.makedirs(dirpath, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create directory {dirpath}: {e}")