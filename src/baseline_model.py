from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# Load frozen training and validation datasets
train = pd.read_csv("data/processed/train.csv")
validation = pd.read_csv("data/processed/validation.csv")

# Exclude mass and identifiers to reduce leakage
FEATURES = [
    "lead_photon_pt_GeV",
    "sublead_photon_pt_GeV",
    "lead_photon_eta",
    "sublead_photon_eta",
    "delta_eta_gg",
    "delta_phi_gg",
    "diphoton_pt_GeV"
]

X_train = train[FEATURES]
y_train = train["label"]

X_val = validation[FEATURES]
y_val = validation["label"]

# Scaling is fitted only on training data
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

# Train the classifier
model.fit(X_train, y_train)

# Evaluate on validation data only
predictions = model.predict(X_val)
scores = model.predict_proba(X_val)[:, 1]

print("Logistic Regression - Validation Results")
print("----------------------------------------")

print("Accuracy:", round(accuracy_score(y_val, predictions), 4))
print("Precision:", round(precision_score(y_val, predictions), 4))
print("Recall:", round(recall_score(y_val, predictions), 4))
print("F1:", round(f1_score(y_val, predictions), 4))
print("ROC-AUC:", round(roc_auc_score(y_val, scores), 4))
print("PR-AUC:", round(average_precision_score(y_val, scores), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, predictions))

print("\nBaseline model completed successfully")