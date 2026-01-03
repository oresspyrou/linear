import pandas as pd
import xgboost as xgb
import numpy as np
import yaml
import sys
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from src.logger_setup import setup_logger
from src.validator import validate_input_file

# Ρύθμιση Logger
try:
    logger = setup_logger()
except RuntimeError as e:
    print(f"CRITICAL: Logger setup failed: {e}")
    sys.exit(1)

def load_config() -> dict:
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
    logger.info("Starting model training (Manual Implementation)...")

    # 1. Load Config & Data
    config = load_config()
    raw_path = config['data']['raw_path']
    validate_input_file(raw_path)

    try:
        df = pd.read_csv(raw_path)
        logger.info(f"Dataset loaded. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)     

    # 2. Manual Cleaning
    # Καθαρισμός κενών και διόρθωση ονομάτων
    df['ocean_proximity'] = df['ocean_proximity'].str.replace(' ', '_')
    
    # Γρήγορο γέμισμα κενών για να τρέξει το μοντέλο (Simple Imputation)
    # Προσοχή: Εδώ γεμίζουμε με 0, αλλά η διάμεσος (median) θα ήταν καλύτερη
    df.fillna(0, inplace=True) 

    # 3. Split Features / Target
    target_col = config['model']['target']
    X = df.drop(columns=[target_col], axis=1)
    y = df[target_col]

    # 4. Train / Test Split
    # ΠΡΟΣΟΧΗ: Κάνουμε το split ΠΡΙΝ το encoding
    test_size = config['model']['test_size']
    random_state = config['model']['random_state']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Split complete. Train: {X_train.shape}, Test: {X_test.shape}")

    # 5. One-Hot Encoding (Η Λογική που ζήτησες)
    logger.info("Encoding categorical variables manually...")
    
    # Αρχικοποίηση Encoder
    # sparse=False: Για να βλέπουμε τα δεδομένα (όχι compressed)
    # drop='first': Για αποφυγή πολυσυγγραμμικότητας (Dummy Variable Trap)
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    
    # Βήμα Α: FIT μόνο στο Train! (Μαθαίνει ποιες κατηγορίες υπάρχουν)
    encoder.fit(X_train[['ocean_proximity']])
    
    # Βήμα Β: TRANSFORM στο Train
    encoded_train = encoder.transform(X_train[['ocean_proximity']])
    encoded_train_df = pd.DataFrame(
        encoded_train, 
        columns=encoder.get_feature_names_out(['ocean_proximity']), 
        index=X_train.index
    )
    # Ενωση (Πετάμε το παλιό, κρατάμε το encoded)
    X_train = pd.concat([X_train.drop('ocean_proximity', axis=1), encoded_train_df], axis=1)
    
    # Βήμα Γ: TRANSFORM στο Test (Χρησιμοποιούμε αυτό που έμαθε από το Train)
    # Αν το Test έχει νέα κατηγορία (π.χ. ISLAND) που δεν υπήρχε στο Train, 
    # θα την αγνοήσει (λόγω handle_unknown='ignore') ή θα βγάλει error.
    encoded_test = encoder.transform(X_test[['ocean_proximity']])
    encoded_test_df = pd.DataFrame(
        encoded_test, 
        columns=encoder.get_feature_names_out(['ocean_proximity']), 
        index=X_test.index
    )
    X_test = pd.concat([X_test.drop('ocean_proximity', axis=1), encoded_test_df], axis=1)

    logger.info("Categorical encoding completed successfully.")

    # 6. Model Training
    logger.info("Initializing XGBoost...")
    # Φόρτωση παραμέτρων από το config
    xgb_params = config['model']['params']
    
    clf_xgb = xgb.XGBRegressor(
        n_estimators=xgb_params['n_estimators'],
        max_depth=xgb_params['max_depth'],
        learning_rate=xgb_params['learning_rate'],
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=random_state
    )

    # Cross Validation (Προαιρετικό αλλά καλό)
    logger.info("Running Cross-Validation on Train set...")
    scores = cross_val_score(clf_xgb, X_train, y_train, cv=5, scoring='r2')
    logger.info(f"Avg Cross-Val R2: {scores.mean():.4f}")

    # Τελική εκπαίδευση
    logger.info("Training final model...")
    clf_xgb.fit(X_train, y_train)

    # 7. Evaluation (External Validation)
    logger.info("Evaluating on Test Set...")
    preds = clf_xgb.predict(X_test)
    
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    logger.info("--- RESULTS ---")
    logger.info(f"R2 Score: {r2:.4f}")
    logger.info(f"RMSE: ${rmse:,.2f}")

    # 8. Save Model & Encoder
    # ΠΡΟΣΟΧΗ: Πρέπει να σώσουμε ΚΑΙ τον encoder, αλλιώς δεν θα μπορούμε να κάνουμε predict σε νέα data!
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(clf_xgb, os.path.join(models_dir, "xgboost_manual.pkl"))
    joblib.dump(encoder, os.path.join(models_dir, "encoder_manual.pkl"))
    logger.info("Model and Encoder saved.")

if __name__ == "__main__":
    train_model()