import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataManager:
    def __init__(self, logger):
        self.logger = logger

    def load_and_clean(self, filepath, target_col):
        """Φορτώνει και καθαρίζει τα βασικά (Strings, Empty values)"""
        self.logger.info(f"Loading data from {filepath}...")
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

        # Manual Cleaning
        if 'ocean_proximity' in df.columns:
            df['ocean_proximity'] = df['ocean_proximity'].str.replace(' ', '_')
        
        # Handling empty strings / NaNs
        # (Εδώ βάζουμε τον βρόχο που είχες στο main)
        for col in df.columns:
            if df[col].dtype == object:
                # Αν υπάρχουν κενά strings, κάντα NaN ή 0
                mask = df[col] == ''
                if mask.any():
                    df.loc[mask, col] = 0 # Ή NaN
            
            # Μετατροπή σε numeric
            df[col] = pd.to_numeric(df[col], errors='ignore')

        df = df.fillna(0) # Simple Imputation για ασφάλεια
        
        # Διαχωρισμός X, y
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        return X, y

    def split_data(self, X, y, test_size, random_state):
        self.logger.info("Splitting data into Train/Test sets...")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)