import os
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class Preprocessor:
    def __init__(self, config, logger):
        """
        Αρχικοποιεί τον Preprocessor.
        """
        self.config = config
        self.logger = logger
        
        self.categorical_cols = self.config['preprocessing']['categorical_columns']
        
        self.logger.info("Encoding categorical variables with OneHotEncoder...")
        self.encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

    def fit_transform(self, X_train):
        """
        Μαθαίνει τις κατηγορίες από το Training set και το μετασχηματίζει.
        """
        self.logger.info("Encoding categorical features (Fit & Transform)...")
        
        missing_cols = [col for col in self.categorical_cols if col not in X_train.columns]
        if missing_cols:
            self.logger.warning(f"Columns {missing_cols} not found in Training data. Skipping encoding for them.")
            valid_cols = [col for col in self.categorical_cols if col in X_train.columns]
        else:
            valid_cols = self.categorical_cols

        if not valid_cols:
            return X_train

        encoded_array = self.encoder.fit_transform(X_train[valid_cols])
        
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=self.encoder.get_feature_names_out(valid_cols),
            index=X_train.index 
        )
        
        X_final = pd.concat([X_train.drop(valid_cols, axis=1), encoded_df], axis=1)
        
        self.logger.info(f"Encoding complete. New shape: {X_final.shape}")
        return X_final

    def transform(self, X_test):
        """
        Μετασχηματίζει το Test set (χωρίς να μάθει).
        """
        self.logger.info("Encoding Test data (Transform only)...")
        
        valid_cols = [col for col in self.categorical_cols if col in X_test.columns]
        
        if not valid_cols:
            return X_test

        encoded_array = self.encoder.transform(X_test[valid_cols])
        
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=self.encoder.get_feature_names_out(valid_cols),
            index=X_test.index 
        )
        
        X_final = pd.concat([X_test.drop(valid_cols, axis=1), encoded_df], axis=1)
        
        return X_final

    def save(self, directory="models", filename="encoder.pkl"):
        """
        Αποθηκεύει τον εκπαιδευμένο encoder για μελλοντική χρήση.
        """
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        
        try:
            joblib.dump(self.encoder, filepath)
            self.logger.info(f"Encoder saved successfully at: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save encoder: {e}")