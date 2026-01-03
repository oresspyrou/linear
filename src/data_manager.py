import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataManager:
    def __init__(self, config, logger):
        """
        Δέχεται το config και τον logger κατά την κατασκευή.
        """
        self.config = config
        self.logger = logger

    def load_and_clean(self):
        """
        Φορτώνει τα δεδομένα και κάνει τον αρχικό καθαρισμό.
        """
        filepath = self.config['data']['raw_path']
        target_col = self.config['model']['target']

        self.logger.info(f"Loading data from {filepath}...")
        
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            self.logger.info("Dataset loaded successfully.")
            self.logger.info(f"Dataset shape: {df.shape}")
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            sys.exit(1) 

        # 1. Καθαρισμός ocean_proximity (αν υπάρχει)
        if 'ocean_proximity' in df.columns:
            df['ocean_proximity'] = df['ocean_proximity'].str.replace(' ', '_')
            df['ocean_proximity'] = df['ocean_proximity'].replace('', 'MISSING').fillna('MISSING')

        # 2. Έλεγχος για κενά strings και μετατροπή σε numeric (όπου πρέπει)
        self.logger.info("Checking for empty strings and fixing data types...")
        
        for col in df.columns:
            # Αγνοούμε την κατηγορική στήλη για να μην την χαλάσουμε
            if col == 'ocean_proximity':
                continue

            if df[col].dtype == object:
                empty_mask = df[col] == ''
                empty_count = len(df[empty_mask])
                
                if empty_count > 0:
                    self.logger.warning(f"Column '{col}' has {empty_count} empty strings.")
                    df.loc[empty_mask, col] = 0
                    self.logger.info(f"Replaced empty strings in column '{col}' with 0.")
                else:
                    self.logger.info(f"Column '{col}' has no empty strings.")
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    self.logger.warning(f"Could not convert column {col} to numeric: {e}")

        nan_counts = df.isna().sum().sum()
        if nan_counts > 0:
             self.logger.info(f"Found {nan_counts} NaN values in total. Filling with 0.")
        

        X = df.drop(columns=[target_col])
        y = df[target_col]

        self.logger.info("Features and target variable separated.")
        self.logger.info(f"Final Features shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y

    def split_data(self, X, y):
        """
        Κάνει το split βάσει των παραμέτρων του config.
        """
        test_size = self.config['model']['test_size']
        random_state = self.config['model']['random_state']

        self.logger.info(f"Splitting data (Test Size: {test_size})...")
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state)