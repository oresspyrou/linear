import pandas as pd
import xgboost as xgb
import numpy as np
import yaml
import sys
import os
import joblib
import optuna  # <--- ΝΕΑ ΠΡΟΣΘΗΚΗ
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from src.logger_setup import setup_logger
from src.validator import validate_input_file

# Setup Logger
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

def objective(trial, X, y):
    """
    Η συνάρτηση που καλεί η Optuna για να βαθμολογήσει έναν συνδυασμό παραμέτρων.
    """
    # 1. Ορίζουμε το "Εύρος Αναζήτησης" (Search Space)
    params = {
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'random_state': 42,
        # Η Optuna διαλέγει τιμές από εδώ:
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2), # Ξεκινάμε με "γρήγορο" LR
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),  # L1 Regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10) # L2 Regularization
    }

    # 2. Τρέχουμε Cross-Validation για να δούμε πόσο καλά τα πάει
    dtrain = xgb.DMatrix(X, label=y)
    
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=1000,
        nfold=3,                    # 3-Fold για ταχύτητα κατά την αναζήτηση
        metrics='rmse',
        early_stopping_rounds=50,
        seed=42,
        verbose_eval=False
    )
    
    # Επιστρέφουμε το καλύτερο RMSE που πέτυχε αυτός ο συνδυασμός
    return cv_results['test-rmse-mean'].min()

def train_model() -> None:
    logger.info("Starting model training (Hybrid Pro Approach)...")
    config = load_config()

    # --- LOADING & CLEANING (ΙΔΙΟ ΜΕ ΠΡΙΝ) ---
    raw_data_path = config['data']['raw_path']
    validate_input_file(raw_data_path)
    df = pd.read_csv(raw_data_path, encoding='utf-8')

    df['ocean_proximity'].replace(' ', '_', regex = True, inplace=True)
    
    # Χειρισμός κενών (όπως τον είχες)
    for col in df.columns:
        if len(df.loc[df[col] == '']) > 0:
            df.loc[df[col] == '', col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df.drop(columns=[config['model']['target']], axis=1)
    y = df[config['model']['target']]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['model']['test_size'], 
        random_state=config['model']['random_state']
    )

    # Encoding (Manual - Όπως το είχες)
    logger.info("Encoding categorical variables...")
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    encoder.fit(X_train[['ocean_proximity']])
    
    encoded_train = encoder.transform(X_train[['ocean_proximity']])
    encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(['ocean_proximity']), index=X_train.index)
    X_train = pd.concat([X_train.drop('ocean_proximity', axis=1), encoded_train_df], axis=1)
    
    encoded_test = encoder.transform(X_test[['ocean_proximity']])
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(['ocean_proximity']), index=X_test.index)
    X_test = pd.concat([X_test.drop('ocean_proximity', axis=1), encoded_test_df], axis=1)

    logger.info("Preprocessing done. Starting Optimization Phases.")

    # --------------------------------------------------------------------------------------------------
    # ΦΑΣΗ 1: OPTUNA (Βρίσκουμε τη δομή του μοντέλου)
    # --------------------------------------------------------------------------------------------------
    logger.info("🧠 PHASE 1: Searching for best structure with Optuna...")
    
    study = optuna.create_study(direction='minimize')
    # Τρέχουμε 20 δοκιμές (Trials). Μπορείς να το αυξήσεις σε 50 αν έχεις χρόνο.
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=20)
    
    best_params = study.best_params
    logger.info(f"✨ Phase 1 Complete. Best Params: {best_params}")

    # --------------------------------------------------------------------------------------------------
    # ΦΑΣΗ 2: REFINEMENT (Ραφινάρισμα με χαμηλό Learning Rate)
    # --------------------------------------------------------------------------------------------------
    logger.info("💎 PHASE 2: Refining with Low Learning Rate (0.01)...")
    
    # 1. Παίρνουμε τις καλές παραμέτρους
    final_params = best_params.copy()
    final_params['objective'] = 'reg:squarederror'
    final_params['n_jobs'] = -1
    
    # 2. Εφαρμόζουμε τον "Χρυσό Κανόνα": Χαμηλώνουμε το Learning Rate
    final_params['learning_rate'] = 0.01 
    
    # 3. Ξανατρέχουμε CV για να βρούμε τα ΝΕΑ δέντρα (θα είναι περισσότερα τώρα)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    logger.info("⏳ Calculating optimal trees for slow learning rate...")
    cv_results = xgb.cv(
        final_params,
        dtrain,
        num_boost_round=5000,       # Δίνουμε μεγάλο περιθώριο
        nfold=5,                    # Εδώ κάνουμε 5-fold για μέγιστη αξιοπιστία
        metrics='rmse',
        early_stopping_rounds=50,
        seed=42,
        verbose_eval=False
    )
    
    optimal_trees = cv_results.shape[0]
    logger.info(f"✅ Optimal Trees found for Low LR: {optimal_trees}")

    # --------------------------------------------------------------------------------------------------
    # ΦΑΣΗ 3: FINAL TRAINING
    # --------------------------------------------------------------------------------------------------
    logger.info("🏋️ PHASE 3: Training Final Model...")
    
    clf_xgb = xgb.XGBRegressor(
        **final_params,
        n_estimators=optimal_trees, # Το νούμερο που βρήκαμε στη Φάση 2
        random_state=42
    )
    
    clf_xgb.fit(X_train, y_train)
    # --------------------------------------------------------------------------------------------------

    logger.info("Evaluating on Test Set...")
    preds = clf_xgb.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    logger.info("--- 🏁 FINAL RESULTS ---")
    logger.info(f"R2 Score: {r2:.4f}")
    logger.info(f"RMSE: ${rmse:,.2f}")
    
    # Save Model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(clf_xgb, os.path.join(models_dir, "xgboost_optimized.pkl"))
    logger.info("Model saved.")

if __name__ == "__main__":
    train_model()