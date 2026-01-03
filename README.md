# California Housing Prediction Pipeline (MLOps Ready)

Ένα ολοκληρωμένο Machine Learning Pipeline για την πρόβλεψη τιμών κατοικιών στην Καλιφόρνια. Το project ακολουθεί αυστηρές αρχές **Separation of Concerns** και ενσωματώνει πλήρως πρακτικές **MLOps** χρησιμοποιώντας **MLflow** για Experiment Tracking και Model Registry, καθώς και **Optuna** για αυτοματοποιημένο Hyperparameter Tuning.

# Βασικά Χαρακτηριστικά

* **Modular Architecture**: Ο κώδικας είναι διαχωρισμένος σε διακριτά modules (`DataManager`, `Preprocessor`, `Optimizer`, `Trainer`) για εύκολη συντήρηση και επεκτασιμότητα.
* **Automated Tuning**: Χρήση της **Optuna** για την εύρεση των βέλτιστων υπερ-παραμέτρων του XGBoost, με τα αποτελέσματα να καταγράφονται αυτόματα.
* **MLflow Integration**: Πλήρης καταγραφή πειραμάτων (Parameters, Metrics, Artifacts, Models) και Versioning των παραγόμενων μοντέλων.
* **Robust Preprocessing**: Αυτόματος χειρισμός missing values και One-Hot Encoding με αποθήκευση του Encoder για χρήση σε Production περιβάλλον.
* **Advanced Training Strategy**: Στρατηγική εκπαίδευσης δύο φάσεων (Optimization -> Retraining) με χαμηλό Learning Rate και Early Stopping για μεγιστοποίηση της απόδοσης.

# Αποτελέσματα

Το μοντέλο πέτυχε εξαιρετική απόδοση στο Test Set:

* **R² Score**: ~0.85 (Επεξήγηση του 85% της διακύμανσης των τιμών).
* **RMSE**: ~$44,300.
* **Top Features**: Τοποθεσία (`INLAND`) και Εισόδημα (`median_income`), επιβεβαιώνοντας την εγκυρότητα του μοντέλου βάσει της οικονομικής θεωρίας.

# Εγκατάσταση

Το project απαιτεί **Python 3.12+**.

# Κλωνοποίηση του Repository
```bash
git clone(https://github.com/oresspyrou/linear.git)
cd california-housing-project