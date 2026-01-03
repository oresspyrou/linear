import xgboost as xgb
import joblib
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import mlflow

class XGBoostTrainer:
    def __init__(self, config, logger):
        """
        Αρχικοποιεί τον εκπαιδευτή του μοντέλου.
        """
        self.config = config
        self.logger = logger
        self.model = None
        self.optimal_trees = 0

    def train_final_model(self, X_train, y_train, best_params):
        """
        Εκτελεί τη Φάση 2 (Refinement) και τη Φάση 3 (Final Training).
        """
        self.logger.info("PHASE 2 & 3: Final Model Construction Started")

        final_params = best_params.copy()
        
        final_params['objective'] = 'reg:squarederror'
        final_params['n_jobs'] = -1
        final_params['verbosity'] = 1
        final_params['random_state'] = self.config['model']['random_state']
        
        final_params['learning_rate'] = 0.01 
        self.logger.info("Learning Rate set to 0.01 for refined training.")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        self.logger.info("Calculating optimal trees with low learning rate...")
        
        
        cv_results = xgb.cv(
            final_params,
            dtrain,
            num_boost_round=10000,      
            nfold=5,                   
            metrics='rmse',
            early_stopping_rounds=50,  
            seed=self.config['model']['random_state'],
            verbose_eval=False
        )
        
        self.optimal_trees = cv_results.shape[0]
        self.logger.info(f"Optimal Trees found: {self.optimal_trees}")

        mlflow.xgboost.autolog(importance_types=['weight', 'gain'])

        self.logger.info(f"Training final model on ALL training data with {self.optimal_trees} trees...")
        
        self.model = xgb.XGBRegressor(
            **final_params,
            n_estimators=self.optimal_trees,
        )
        
        self.model.fit(X_train, y_train)
        self.logger.info("Final Model Training Complete.")

    def evaluate(self, X_test, y_test):
        """
        Αξιολογεί το μοντέλο στο Test set και επιστρέφει τα αποτελέσματα.
        """
        if self.model is None:
            self.logger.warning("Model is not trained yet!")
            return {}

        self.logger.info("Evaluating on Test Set...")
        preds = self.model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        self.logger.info("--- FINAL RESULTS ---")
        self.logger.info(f"RMSE: {rmse:,.4f}")
        self.logger.info(f"R2 Score: {r2:.4f}")

        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)

        return {'rmse': rmse, 'r2': r2}

    def save_model(self, filename="xgboost_final.pkl"):
        """
        Αποθηκεύει το μοντέλο τοπικά και το καταχωρεί στο MLflow Registry.
        """
        if self.model is None:
            self.logger.warning("No model to save.")
            return

        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        filepath = os.path.join(models_dir, filename)
        
        try:
            joblib.dump(self.model, filepath)
            self.logger.info(f"Model saved locally to: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save local model: {e}")

        self.logger.info("Registering model to MLflow...")
        
        try:
            mlflow.xgboost.log_model(
                xgb_model=self.model, 
                artifact_path="model", 
                registered_model_name="CaliforniaHousing_Production_Model"
            )
            self.logger.info("Model successfully registered in MLflow.")
            
        except Exception as e:
            self.logger.warning(f"Failed to register model to MLflow: {e}")