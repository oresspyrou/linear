import xgboost as xgb
import optuna
import sys

class ModelOptimizer:
    def __init__(self, config, logger):
        """
        Αρχικοποιεί τον Optimizer με το config και τον logger.
        """
        self.config = config
        self.logger = logger

    def objective(self, trial, X, y):
        """
        Η συνάρτηση που καλεί η Optuna σε κάθε 'trial' (δοκιμή).
        """
        space = self.config['optimization']['search_space']
        
        # 2. Βασικές παράμετροι (σταθερές)
        params = {
            'objective': 'reg:squarederror',
            'n_jobs': -1,
            'random_state': self.config['model']['random_state'],
            'verbosity': 0  # Για να μην γεμίζει το log με μηνύματα του XGBoost
        }

        # 3. Δυναμική επιλογή παραμέτρων (το "Έξυπνο Loop" που φτιάξαμε)
        for param_name, bounds in space.items():
            # Ελέγχουμε αν τα όρια είναι int ή float
            is_int = isinstance(bounds['low'], int) and isinstance(bounds['high'], int)
            
            if is_int:
                # Αν είναι int (π.χ. max_depth), χρησιμοποιούμε suggest_int
                # Το **bounds περνάει τα low, high (και step/log αν υπάρχουν)
                params[param_name] = trial.suggest_int(param_name, **bounds)
            else:
                # Αν είναι float (π.χ. learning_rate), χρησιμοποιούμε suggest_float
                params[param_name] = trial.suggest_float(param_name, **bounds)

        # 4. Εκτέλεση Cross-Validation
        # Φτιάχνουμε το DMatrix (ειδική δομή του XGBoost για ταχύτητα)
        dtrain = xgb.DMatrix(X, label=y)
        
        try:
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=1000,
                nfold=3,                    # 3-Fold για ταχύτητα στην αναζήτηση
                metrics='rmse',
                early_stopping_rounds=50,
                seed=self.config['model']['random_state'],
                verbose_eval=False          # Δεν θέλουμε prints σε κάθε βήμα
            )
            
            # Επιστρέφουμε το καλύτερο (χαμηλότερο) RMSE που βρέθηκε
            return cv_results['test-rmse-mean'].min()
            
        except Exception as e:
            # Αν κάτι σκάσει (π.χ. πολύ κακός συνδυασμός παραμέτρων), επιστρέφουμε ένα τεράστιο λάθος
            # ώστε η Optuna να αποφύγει αυτόν τον δρόμο.
            self.logger.warning(f"Trial failed with error: {e}")
            return float('inf')

    def run_optimization(self, X, y):
        """
        Εκκινεί τη διαδικασία της Optuna (Phase 1).
        """
        n_trials = self.config['optimization']['n_trials']
        self.logger.info(f"🧠 PHASE 1: Starting Optuna Optimization ({n_trials} trials)...")

        # Δημιουργία Study
        # direction='minimize' -> Θέλουμε να ελαχιστοποιήσουμε το RMSE
        study = optuna.create_study(direction='minimize')

        # Εκτέλεση Study
        # Χρησιμοποιούμε lambda για να περάσουμε το self.objective με τα X, y
        study.optimize(lambda trial: self.objective(trial, X, y), n_trials=n_trials)

        self.logger.info("✨ Phase 1 Complete.")
        self.logger.info(f"Best Score (RMSE): {study.best_value:.4f}")
        self.logger.info(f"Best Params found: {study.best_params}")

        return study.best_params