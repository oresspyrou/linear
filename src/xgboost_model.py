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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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

def train_model() -> None:
    """
    Train the XGBoost model on the California housing dataset.
    
    Loads configuration, validates data files, loads the dataset,
    separates features and target, and prepares for model training.
    """
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
        df = pd.read_csv(raw_data_path, encoding='utf-8')
        logger.info("Dataset loaded successfully.")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Dataset preview:\n{df.head()}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)     

    df['ocean_proximity'].replace(' ', '_', regex = True, inplace=True)
    df.dtypes

    logger.info("Checking for empty strings in each column...")
    for col in df.columns:
        empty_count = len(df.loc[df[col] == ''])
        if empty_count > 0:
            logger.warning(f"Column '{col}' has {empty_count} empty strings.")
            df.loc[df[col] == '', col] = 0
            logger.info(f"Replaced empty strings in column '{col}' with 0.")
        else:
            logger.info(f"Column '{col}' has no empty strings.")

    logger.info("Dropping the target column from features")
    x = df.drop(columns=[config['model']['target']], axis=1)
    y = df[config['model']['target']]
    logger.info("Features and target variable separated.")
    logger.info(f"Features shape: {x.shape}, Target shape: {y.shape}")
    logger.info(f"Features preview:\n{x.head()}")

    logger.info(f"Unique values in 'ocean_proximity': {x['ocean_proximity'].unique()}")

    logger.info("Encoding categorical variables with OneHotEncoder...")
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded = encoder.fit_transform(x[['ocean_proximity']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['ocean_proximity']), index=x.index)
    x = pd.concat([x.drop('ocean_proximity', axis=1), encoded_df], axis=1)
    logger.info("Categorical encoding completed.")
    logger.info(f"New features after encoding: {[col for col in x.columns if 'ocean_proximity' in col]}")
    logger.info(f"Features shape after encoding: {x.shape}")

    logger.info("Splitting dataset into training and testing sets...")
    test_size = config['model']['test_size']
    random_state = config['model']['random_state']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)