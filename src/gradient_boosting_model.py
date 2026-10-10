import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# Load the same frozen datasets
train = pd.read_csv("data/processed/train.csv")
validation = pd.read_csv("data/processed/validation.csv")

# Same feature set as previous models
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

# Gradient Boosting model
model = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=20,
    random_state=42
)

# Train only on training data
model.fit(X_train, y_train)

# Evaluate only on validation data
predictions = model.predict(X_val)
scores = model.predict_proba(X_val)[:, 1]

print("Gradient Boosting - Validation Results")
print("--------------------------------------")

print("Accuracy:", round(accuracy_score(y_val, predictions), 4))
print("Precision:", round(precision_score(y_val, predictions), 4))
print("Recall:", round(recall_score(y_val, predictions), 4))
print("F1:", round(f1_score(y_val, predictions), 4))
print("ROC-AUC:", round(roc_auc_score(y_val, scores), 4))
print("PR-AUC:", round(average_precision_score(y_val, scores), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, predictions))

print("\nGradient Boosting completed successfully")

# Evaluate performance in the Higgs mass region
mass_mask = validation["diphoton_mass_GeV"].between(
    120, 130
)

y_region = y_val[mass_mask]
pred_region = predictions[mass_mask]
scores_region = scores[mass_mask]

print("\nHiggs mass region (120-130 GeV)")
print("--------------------------------")

print("Events:", len(y_region))
print("Class counts:")
print(y_region.value_counts().sort_index())

print("\nAccuracy:", round(
    accuracy_score(y_region, pred_region), 4
))

print("Precision:", round(
    precision_score(y_region, pred_region, zero_division=0), 4
))

print("Recall:", round(
    recall_score(y_region, pred_region, zero_division=0), 4
))

print("F1:", round(
    f1_score(y_region, pred_region, zero_division=0), 4
))

if y_region.nunique() == 2:
    print("ROC-AUC:", round(
        roc_auc_score(y_region, scores_region), 4
    ))

    print("PR-AUC:", round(
        average_precision_score(y_region, scores_region), 4
    ))

print("\nConfusion Matrix:")
print(confusion_matrix(y_region, pred_region, labels=[0, 1]))