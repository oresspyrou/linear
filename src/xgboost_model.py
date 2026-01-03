import pandas as pd
import xgboost as xgb
import numpy as np
import yaml
import sys
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, balanced_accuracy_score, roc_auc_score, make_scorer, confusion_matrix 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from src.logger_setup import setup_logger
from src.validator import validate_input_file, validate_csv_columns
from sklearn.model_selection import KFold, cross_val_score
import optuna

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

#---------------------------------------------------------------------------------------------

def objective(trial, X, y, config):
    """
    Η συνάρτηση που καλεί η Optuna για να βαθμολογήσει έναν συνδυασμό παραμέτρων.
    """
    space = config['optimization']['search_space']
    
    params = {
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'random_state': config['model']['random_state'],
    }

    for param_name, bounds in space.items():
        is_int = isinstance(bounds['low'], int) and isinstance(bounds['high'], int)
        
        if is_int:
            params[param_name] = trial.suggest_int(param_name, **bounds)
        else:
            params[param_name] = trial.suggest_float(param_name, **bounds)

    dtrain = xgb.DMatrix(X, label=y)
    
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=1000,
        nfold=3,                    
        metrics='rmse',
        early_stopping_rounds=50,
        seed=42,
        verbose_eval=True
    )

    return cv_results['test-rmse-mean'].min()

#---------------------------------------------------------------------------------------------  

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

            nan_count = df[col].isna().sum()
            logger.info(f"Column '{col}' has {nan_count} NaN values after replacement.")

            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dtypes
        else:
            logger.info(f"Column '{col}' has no empty strings.")

    logger.info("Dropping the target column from features")
    X = df.drop(columns=[config['model']['target']], axis=1)
    y = df[config['model']['target']]
    logger.info("Features and target variable separated.")
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    logger.info(f"Features preview:\n{X.head()}")
    logger.info(f"Unique values in 'ocean_proximity': {X['ocean_proximity'].unique()}")

    logger.info("Splitting dataset into training and testing sets...")
    test_size = config['model']['test_size']
    random_state = config['model']['random_state']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    logger.info("Encoding categorical variables with OneHotEncoder...")
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoder.fit(X_train[['ocean_proximity']])
   
    encoded_train = encoder.transform(X_train[['ocean_proximity']])
    encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(['ocean_proximity']), index=X_train.index)
    X_train = pd.concat([X_train.drop('ocean_proximity', axis=1), encoded_train_df], axis=1)
    
    encoded_test = encoder.transform(X_test[['ocean_proximity']])
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(['ocean_proximity']), index=X_test.index)
    X_test = pd.concat([X_test.drop('ocean_proximity', axis=1), encoded_test_df], axis=1)

    logger.info("Categorical encoding completed.")
    logger.info(f"New features after encoding: {[col for col in X_train.columns if 'ocean_proximity' in col]}")
    logger.info(f"Training features shape: {X_train.shape}, Test features shape: {X_test.shape}")

    logger.info(f"Unique values in target variable: {y.unique()}")

    logger.info("Model training setup completed. Ready for training phase.")
    #---------------------------------------------------------------------------------------------------------------------------------------------
    logger.info("Running Internal XGBoost CV to find optimal trees...")
    logger.info("PHASE 1: Searching for best structure with Optuna...")

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, config), n_trials=20)

    best_params = study.best_params
    logger.info(f"Phase 1 Complete. Best Params: {best_params}")

    logger.info("PHASE 2: Refining with Low Learning Rate (0.01)...")

    final_params = best_params.copy()
    final_params['objective'] = 'reg:squarederror'
    final_params['n_jobs'] = -1
    final_params['learning_rate'] = 0.01

    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    logger.info("Calculating optimal trees for slow learning rate...")
    cv_results = xgb.cv(
        final_params,
        dtrain,
        num_boost_round=5000,       
        nfold=5,                    
        metrics='rmse',
        early_stopping_rounds=50,
        seed=42,
        verbose_eval=False
    )

    optimal_trees = cv_results.shape[0] 
    logger.info(f"Smart Training determined optimal n_estimators: {optimal_trees}")

    logger.info("Training final model with optimal trees...")

    clf_xgb = xgb.XGBRegressor(
        **final_params,
        n_estimators=optimal_trees,
        random_state=random_state,
        objective='reg:squarederror'
    )

    clf_xgb.fit(X_train, y_train)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------

    logger.info("Evaluating on Test Set...")
    preds = clf_xgb.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    logger.info("--- RESULTS ---")
    logger.info(f"R2 Score: {r2:.4f}")
    logger.info(f"RMSE: ${rmse:,.2f}")

    if __name__ == "__main__":
        train_model()
