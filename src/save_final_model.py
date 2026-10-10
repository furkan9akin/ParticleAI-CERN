from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier

RANDOM_SEED = 42

FEATURES = [
    "lead_photon_pt_GeV",
    "sublead_photon_pt_GeV",
    "lead_photon_eta",
    "sublead_photon_eta",
    "delta_eta_gg",
    "delta_phi_gg",
    "diphoton_pt_GeV"
]

# Use training data only
train = pd.read_csv("data/processed/train.csv")

X_train = train[FEATURES]
y_train = train["label"]

# Frozen configuration selected using validation results
model = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=20,
    random_state=RANDOM_SEED
)

model.fit(X_train, y_train)

# Save model locally
output = Path("models")
output.mkdir(exist_ok=True)

model_path = output / "gradient_boosting_v1.joblib"
joblib.dump(model, model_path)

# Save reproducibility metadata
metadata = {
    "model": "HistGradientBoostingClassifier",
    "version": "v1",
    "random_seed": RANDOM_SEED,
    "features": FEATURES,
    "parameters": model.get_params(),
    "training_events": len(train),
    "signal_sample": "343981",
    "background_sample": "364352",
    "mass_used_as_feature": False,
    "classification_threshold": 0.5,
    "evaluation_status": "Validation completed; test pending"
}

with open(
    output / "gradient_boosting_v1_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=2)

print("Model saved:", model_path)
print("Training events:", len(train))
print("Features:", len(FEATURES))
print("Model configuration frozen")