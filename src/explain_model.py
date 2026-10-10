from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

train = pd.read_csv("data/processed/train.csv")
validation = pd.read_csv("data/processed/validation.csv")

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

# Same Gradient Boosting configuration
model = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=20,
    random_state=42
)

model.fit(X_train, y_train)

# Measure permutation importance using ROC-AUC
result = permutation_importance(
    model,
    X_val,
    y_val,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": result.importances_mean,
    "std": result.importances_std
}).sort_values("importance", ascending=False)

print("\nPermutation Importance (ROC-AUC decrease)")
print(importance.to_string(index=False))

# Create bar chart
plt.figure(figsize=(9, 5))

plt.barh(
    importance["feature"],
    importance["importance"],
    xerr=importance["std"]
)

plt.gca().invert_yaxis()

plt.xlabel("Mean decrease in ROC-AUC")
plt.title("Gradient Boosting - Permutation Feature Importance")

plt.tight_layout()

Path("figures").mkdir(exist_ok=True)

plt.savefig(
    "figures/feature_importance.png",
    dpi=150
)

plt.show()