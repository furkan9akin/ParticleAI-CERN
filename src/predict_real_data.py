from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load previously frozen model
model = joblib.load(
    "models/gradient_boosting_v1.joblib"
)

with open(
    "models/gradient_boosting_v1_metadata.json",
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)

# Load real ATLAS collision data
real_data = pd.read_csv(
    "data/processed/real_data_processed.csv"
)

# Use exactly the same seven features as training
features = metadata["features"]
X_real = real_data[features]

# Basic data checks
assert np.all(np.isfinite(X_real.to_numpy()))

# Apply frozen classifier
scores = model.predict_proba(X_real)[:, 1]

print("Real ATLAS Data - Model Scores")
print("--------------------------------")
print("Events:", len(scores))
print("Mean score:", round(float(scores.mean()), 4))
print("Median score:", round(float(np.median(scores)), 4))
print("Minimum score:", round(float(scores.min()), 4))
print("Maximum score:", round(float(scores.max()), 4))

print("\nScore distribution:")
print("Score below 0.5:", int((scores < 0.5).sum()))
print("Score >= 0.5:", int((scores >= 0.5).sum()))
print("Score >= 0.8:", int((scores >= 0.8).sum()))

# Save scores locally
real_data["signal_score"] = scores

output = Path("data/processed")
output.mkdir(parents=True, exist_ok=True)

real_data.to_csv(
    output / "real_data_scores.csv",
    index=False
)

# Plot distribution
plt.figure(figsize=(8, 5))

plt.hist(
    scores,
    bins=30,
    range=(0, 1),
    edgecolor="black"
)

plt.xlabel("Model signal-likeness score")
plt.ylabel("Number of real collision events")
plt.title("Gradient Boosting Scores - Real ATLAS Data")

plt.tight_layout()

Path("figures").mkdir(exist_ok=True)

plt.savefig(
    "figures/real_data_scores.png",
    dpi=150
)

plt.show()

print("\nReal data scoring completed")