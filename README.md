# 🏡 California Housing Prediction Pipeline (MLOps Ready)

A comprehensive Machine Learning Pipeline for predicting housing prices in California. This project adheres to strict **Separation of Concerns** principles and fully integrates **MLOps** best practices using **MLflow** for Experiment Tracking and Model Registry, as well as **Optuna** for automated Hyperparameter Tuning.

## 🚀 Key Features

* **Modular Architecture**: Code is organized into distinct modules (`DataManager`, `Preprocessor`, `Optimizer`, `Trainer`) for maintainability and scalability.
* **Automated Tuning**: Utilizes **Optuna** to find optimal XGBoost hyperparameters, with automatic logging of all trials.
* **MLflow Integration**: Full experiment tracking (Parameters, Metrics, Artifacts, Models) and versioning of production-ready models.
* **Robust Preprocessing**: Automatic handling of missing values and One-Hot Encoding, preserving the Encoder as an artifact for production inference.
* **Advanced Training Strategy**: Two-phase training strategy (Optimization -> Retraining) with low Learning Rate and Early Stopping to maximize performance.

## 📊 Results

The model achieved excellent performance on the Test Set:

* **R² Score**: ~0.85 (Explains 85% of price variance).
* **RMSE**: ~$44,300.
* **Top Features**: Location (`INLAND`) and Income (`median_income`), confirming the model's alignment with economic theory.

## 🛠️ Installation

The project requires **Python 3.12+**.

### 1. Clone Repository
```bash
git clone(https://github.com/oresspyrou/Xgboost-housing-valuation-pipeline.git)
cd california-housing-project
