import pandas as pd
import xgboost as xgb
import numpy as np
import yaml
import sys
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, balanced_accuracy_score, roc_auc_score, make_scorer, confusion_matrix, plot_confusion_matrix  
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from src.logger_setup import setup_logger
from src.validator import validate_input_file, validate_csv_columns

"""
XGBoost Model Module

This module handles the training, evaluation, and configuration loading for XGBoost models
on the California housing dataset. It includes data loading, model training with hyperparameter
tuning, and performance metrics calculation.
"""

try:
    logger = setup_logger()
except RuntimeError as e:
    print(f"CRITICAL: Logger setup failed: {e}")
    sys.exit(1)

def load_config() -> dict:
    """
    Load configuration settings from the YAML config file.
    
    Returns:
        dict: Configuration dictionary loaded from config.yaml.
        
    Raises:
        SystemExit: If the config file cannot be loaded or parsed.
    """
    config_path = "config/config.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        logger.info(f"Config loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

def train_model():
    logger.info("Starting model training...")

    config = load_config()
    
    raw_data_path = config['data']['raw_path']
    validate_input_file(raw_data_path)
    logger.info(f"Validated input file at {raw_data_path}")

    processed_data_path = config['data']['processed_path']
    validate_input_file(processed_data_path)
    logger.info(f"Validated input file at {processed_data_path}")

    logger.info("Loading dataset...")

    try:
        df = pd.read_csv(processed_data_path)
        logger.info("Dataset loaded successfully.")
        logger.info("Dataset preview:")
        df.head()
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)     

    