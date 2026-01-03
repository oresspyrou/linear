🏡 California Housing Prediction Pipeline (MLOps Ready)
Ένα ολοκληρωμένο Machine Learning Pipeline για την πρόβλεψη τιμών κατοικιών στην Καλιφόρνια. Το project ακολουθεί αυστηρές αρχές Separation of Concerns και ενσωματώνει πλήρως πρακτικές MLOps χρησιμοποιώντας MLflow για Experiment Tracking και Model Registry, καθώς και Optuna για αυτοματοποιημένο Hyperparameter Tuning.

🚀 Βασικά Χαρακτηριστικά
Modular Architecture: Ο κώδικας είναι διαχωρισμένος σε διακριτά modules (DataManager, Preprocessor, Optimizer, Trainer) για εύκολη συντήρηση.

Automated Tuning: Χρήση της Optuna για την εύρεση των βέλτιστων υπερ-παραμέτρων του XGBoost.

MLflow Integration: Πλήρης καταγραφή πειραμάτων (Parameters, Metrics, Artifacts, Models) και Versioning.

Robust Preprocessing: Αυτόματος χειρισμός missing values και One-Hot Encoding με αποθήκευση του Encoder για χρήση σε Production.

Advanced Training Strategy: Στρατηγική εκπαίδευσης με χαμηλό Learning Rate και Early Stopping για μεγιστοποίηση της απόδοσης.

📊 Αποτελέσματα
Το μοντέλο πέτυχε εξαιρετική απόδοση στο Test Set:

R² Score: ~0.85 (Επεξήγηση του 85% της διακύμανσης των τιμών).

RMSE: ~$44,300.

Top Features: Τοποθεσία (INLAND) και Εισόδημα (median_income), όπως αναμενόταν από την οικονομική θεωρία.

🛠️ Εγκατάσταση
Το project απαιτεί Python 3.12+.

1. Κλωνοποίηση
Bash

git clone <repository-url>
cd california-housing-project
2. Εγκατάσταση Εξαρτήσεων
Προτείνεται η χρήση του uv (όπως έχει ρυθμιστεί στο project), αλλά υποστηρίζεται και το κλασικό pip.

Με uv (Προτείνεται):

Bash

uv sync
Με pip:

Bash

pip install .
Οι κύριες βιβλιοθήκες περιλαμβάνουν: xgboost, mlflow, optuna, pandas, scikit-learn.

⚙️ Ρύθμιση (Configuration)
Όλες οι παράμετροι του pipeline βρίσκονται στο αρχείο config/config.yaml:

Data paths: Διαδρομές για τα raw και processed δεδομένα.

Optimization: Εύρος αναζήτησης (Search Space) για την Optuna.

MLflow: Ορισμός του Experiment Name και Tracking URI.

▶️ Εκτέλεση (Usage)
Για να τρέξει ολόκληρο το pipeline (από τη φόρτωση δεδομένων μέχρι την αποθήκευση του μοντέλου):

Bash

python main.py
🖥️ Παρακολούθηση με MLflow
Μετά (ή κατά τη διάρκεια) της εκτέλεσης, μπορείτε να δείτε τα αποτελέσματα στο UI του MLflow:

Bash

mlflow ui
Ανοίξτε τον browser στο http://127.0.0.1:5000.

🏗️ Αρχιτεκτονική Pipeline
Η ροή εκτέλεσης (main.py) ακολουθεί τα εξής βήματα:

Step A - Data Loading: Φόρτωση και καθαρισμός δεδομένων (χειρισμός NaN).

Step B - Preprocessing: One-Hot Encoding κατηγορικών μεταβλητών (ocean_proximity) και αποθήκευση του encoder.pkl ως artifact.

Step C - Hyperparameter Optimization: Εκτέλεση της Optuna (Phase 1) για εύρεση των βέλτιστων παραμέτρων. Κάθε trial καταγράφεται ως nested run στο MLflow.

Step D - Final Training: Εκπαίδευση του τελικού μοντέλου XGBoost με τις βέλτιστες παραμέτρους, μειωμένο Learning Rate (0.01) και αυξημένα δέντρα για μέγιστη ακρίβεια.

Step E - Evaluation: Υπολογισμός μετρικών (RMSE, R2) στο Test Set.

Step F - Registry: Αποθήκευση του μοντέλου (.pkl) και καταχώρηση στο MLflow Model Registry για version control.

📂 Δομή Φακέλων
Plaintext

california-housing-project/
├── config/
│   └── config.yaml       # Κεντρικές ρυθμίσεις
├── data/
│   └── raw/              # Αρχικά δεδομένα (housing.csv)
├── models/               # Αποθηκευμένα μοντέλα & encoders
├── mlruns/               # Τοπική βάση δεδομένων του MLflow
├── src/                  # Πηγαίος κώδικας
│   ├── data_manager.py   # Φόρτωση & Split δεδομένων
│   ├── preprocessing.py  # Feature Engineering & Encoding
│   ├── optimisation.py   # Optuna logic
│   ├── xgb.py            # XGBoost training logic
│   ├── logger_setup.py   # Ρυθμίσεις logging
│   └── utils.py          # Βοηθητικές συναρτήσεις
├── main.py               # Το entry point του pipeline
├── pyproject.toml        # Dependencies & Project metadata
└── README.md             # Documentation
📝 License
This project is licensed under the MIT License.