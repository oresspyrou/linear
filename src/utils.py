import yaml
import sys
import os

def load_config(config_path="config/config.yaml") -> dict:
    """
    Φορτώνει τις ρυθμίσεις από το αρχείο YAML.
    
    Λειτουργεί με τη λογική 'Fail Fast': Αν δεν βρει το αρχείο ή αν το YAML
    είναι κατεστραμμένο, τερματίζει το πρόγραμμα αμέσως (sys.exit), 
    ώστε να μην τρέχουμε μοντέλα με λάθος ρυθμίσεις.

    Args:
        config_path (str): Η διαδρομή προς το αρχείο config.yaml.

    Returns:
        dict: Το λεξικό με τις ρυθμίσεις.
    """
    # 1. Έλεγχος ύπαρξης αρχείου
    if not os.path.exists(config_path):
        print(f"CRITICAL ERROR: Config file not found at: {config_path}")
        print("Please ensure 'config/config.yaml' exists.")
        sys.exit(1)

    # 2. Προσπάθεια ανάγνωσης
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        if config is None:
            print(f"CRITICAL ERROR: Config file at {config_path} is empty!")
            sys.exit(1)
            
        return config

    except yaml.YAMLError as e:
        print(f"CRITICAL ERROR: Error parsing YAML file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: Unexpected error loading config: {e}")
        sys.exit(1)