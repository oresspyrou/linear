import xgboost as xgb
import optuna
import sys
import mlflow
from optuna.integration.mlflow import MLflowCallback

class ModelOptimizer:
    def __init__(self, config, logger):
        """
        Αρχικοποιεί τον Optimizer με το config και τον logger.
        """
        self.config = config
        self.logger = logger

    def objective(self, trial, X, y):
        """
        Η συνάρτηση που καλεί η Optuna σε κάθε 'trial'.
        """
        space = self.config['optimization']['search_space']
        
        params = {
            'objective': 'reg:squarederror',
            'n_jobs': -1,
            'random_state': self.config['model']['random_state'],
            'verbosity': 1
        }

        for param_name, bounds in space.items():
            is_int = isinstance(bounds['low'], int) and isinstance(bounds['high'], int)
            
            if is_int:
                params[param_name] = trial.suggest_int(param_name, **bounds)
            else:
                params[param_name] = trial.suggest_float(param_name, **bounds)


        dtrain = xgb.DMatrix(X, label=y)
        
        try:
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=1000,
                nfold=3,                  
                metrics='rmse',
                early_stopping_rounds=50,
                seed=self.config['model']['random_state'],
                verbose_eval=True
            )
            
            # Επιστρέφει το καλύτερο (χαμηλότερο) RMSE που βρέθηκε
            return cv_results['test-rmse-mean'].min()
            
        except Exception as e:
            self.logger.warning(f"Trial failed with error: {e}")
            return float('inf')

    def run_optimization(self, X, y):
        """
        Εκκινεί τη διαδικασία της Optuna (Phase 1).
        """
        n_trials = self.config['optimization']['n_trials']
        self.logger.info(f"PHASE 1: Starting Optuna Optimization ({n_trials} trials)...")
        
        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            metric_name="rmse",
            nest_trials=True  # Σημαντικό: Τα trials θα είναι nested κάτω από το main run
        )

        # direction='minimize' -> Θέλουμε να ελαχιστοποιήσουμε το RMSE
        study = optuna.create_study(direction='minimize')

        # Χρησιμοποιούμε lambda για να περάσουμε το self.objective με τα X, y
        study.optimize(
            lambda trial: self.objective(trial, X, y), 
            n_trials=n_trials,
            callbacks=[mlflow_callback])

        self.logger.info("Phase 1 Complete.")
        self.logger.info(f"Best Score (RMSE): {study.best_value:.4f}")
        self.logger.info(f"Best Params found: {study.best_params}")

        mlflow.log_params(study.best_params)

        return study.best_params