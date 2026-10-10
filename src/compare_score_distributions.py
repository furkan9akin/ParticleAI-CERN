import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load saved model and feature metadata
model = joblib.load("models/gradient_boosting_v1.joblib")

with open(
    "models/gradient_boosting_v1_metadata.json",
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)

features = metadata["features"]

# Use validation MC data, not the final test dataset
mc = pd.read_csv("data/processed/validation.csv")

# Load previously scored real collision data
real = pd.read_csv("data/processed/real_data_scores.csv")

# Model scores for MC samples
mc_scores = model.predict_proba(mc[features])[:, 1]

signal_scores = mc_scores[mc["label"].to_numpy() == 1]
background_scores = mc_scores[mc["label"].to_numpy() == 0]

real_scores = real["signal_score"].to_numpy()

# Plot normalized distributions
plt.figure(figsize=(9, 5))

bins = np.linspace(0, 1, 31)

plt.hist(
    signal_scores,
    bins=bins,
    density=True,
    histtype="step",
    linewidth=2,
    label="Higgs MC signal"
)

plt.hist(
    background_scores,
    bins=bins,
    density=True,
    histtype="step",
    linewidth=2,
    label="Sherpa diphoton MC"
)

plt.hist(
    real_scores,
    bins=bins,
    density=True,
    histtype="step",
    linewidth=2,
    label="Real ATLAS data"
)

plt.xlabel("Model signal-likeness score")
plt.ylabel("Normalized density")
plt.title("Model Score Distributions: MC vs Real ATLAS Data")

plt.legend()
plt.tight_layout()

Path("figures").mkdir(exist_ok=True)

plt.savefig(
    "figures/mc_vs_real_scores.png",
    dpi=150
)

plt.show()

print("\nScore comparison completed")
print("Higgs MC events:", len(signal_scores))
print("Background MC events:", len(background_scores))
print("Real events:", len(real_scores))