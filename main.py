import sys
import mlflow
import os
from src import (
    setup_logger, 
    load_config, 
    DataManager, 
    Preprocessor, 
    ModelOptimizer, 
    XGBoostTrainer
)

def main():
    config = load_config() 
    logger = setup_logger()
    
    logger.info("Pipeline Started")

    try:
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])

        logger.info(f"MLflow Tracking URI: {config['mlflow']['tracking_uri']}")
        logger.info(f"Experiment Name: {config['mlflow']['experiment_name']}")

        with mlflow.start_run(run_name=config['mlflow']['run_name']) as run:
            
            logger.info("--- STEP A: Data Loading & Splitting ---")
            dm = DataManager(config, logger)
            X, y = dm.load_and_clean()
            X_train, X_test, y_train, y_test = dm.split_data(X, y)

            logger.info("--- STEP B: Preprocessing ---")
            preprocessor = Preprocessor(config, logger)
            
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)
            
            preprocessor.save() 
            mlflow.log_artifact("models/encoder.pkl", artifact_path="preprocessor")

            logger.info("--- STEP C: Hyperparameter Optimization ---")
           
            optimizer = ModelOptimizer(config, logger)
            best_params = optimizer.run_optimization(X_train, y_train)

            logger.info("--- STEP D: Final Model Training ---")
        
            trainer = XGBoostTrainer(config, logger)
            trainer.train_final_model(X_train, y_train, best_params)


            logger.info("--- STEP E: Evaluation ---")

            metrics = trainer.evaluate(X_test, y_test)
            mlflow.log_metrics(metrics)
            logger.info(f"Test Metrics: {metrics}")

            logger.info("--- STEP F: Saving & Registry ---")
        
            trainer.save_model()

            logger.info(f"Pipeline finished successfully. Run ID: {run.info.run_id}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
