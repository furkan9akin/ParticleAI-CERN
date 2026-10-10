import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

# Load frozen datasets
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

# Train the same model
model = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=20,
    random_state=42
)

model.fit(train[FEATURES], train["label"])

# Predictions on validation data
predictions = model.predict(validation[FEATURES])
scores = model.predict_proba(validation[FEATURES])[:, 1]

# Create error analysis table
results = validation.copy()
results["prediction"] = predictions
results["signal_score"] = scores

# Identify the four prediction categories
results["category"] = "unknown"

results.loc[
    (results["label"] == 1) &
    (results["prediction"] == 1),
    "category"
] = "TP"

results.loc[
    (results["label"] == 0) &
    (results["prediction"] == 0),
    "category"
] = "TN"

results.loc[
    (results["label"] == 0) &
    (results["prediction"] == 1),
    "category"
] = "FP"

results.loc[
    (results["label"] == 1) &
    (results["prediction"] == 0),
    "category"
] = "FN"

print("Prediction categories:")
print(results["category"].value_counts())

print("\nMean feature values by category:")
print(
    results.groupby("category")[
        [
            "lead_photon_pt_GeV",
            "sublead_photon_pt_GeV",
            "diphoton_pt_GeV",
            "diphoton_mass_GeV",
            "signal_score"
        ]
    ].mean().round(2)
)

print("\nMost confident false positives:")
print(
    results[results["category"] == "FP"]
    .sort_values("signal_score", ascending=False)[
        ["diphoton_mass_GeV", "signal_score"]
    ]
    .head(5)
    .round(3)
)

print("\nMost confident false negatives:")
print(
    results[results["category"] == "FN"]
    .sort_values("signal_score", ascending=True)[
        ["diphoton_mass_GeV", "signal_score"]
    ]
    .head(5)
    .round(3)
)
# Error analysis by diphoton mass region

mass_bins = [0, 100, 120, 130, 150, 200, 1000]

results["mass_region"] = pd.cut(
    results["diphoton_mass_GeV"],
    bins=mass_bins,
    include_lowest=True
)

results["is_error"] = (
    results["label"] != results["prediction"]
)

summary = results.groupby(
    "mass_region",
    observed=True
).agg(
    events=("label", "size"),
    errors=("is_error", "sum")
)

summary["error_rate_percent"] = (
    100 * summary["errors"] / summary["events"]
)

print("\nError analysis by mass region:")
print(summary.round(2))

# Investigate the region with the highest observed error rate
region = results[
    results["diphoton_mass_GeV"].between(
        130, 150,
        inclusive="right"
    )
]

print("\nDetailed analysis: 130-150 GeV")
print("--------------------------------")

print("Total events:", len(region))

print("\nTrue class counts:")
print(region["label"].value_counts().sort_index())

print("\nPrediction categories:")
print(region["category"].value_counts())

print("\nMean photon pT by category:")
print(
    region.groupby("category")[
        ["lead_photon_pt_GeV", "sublead_photon_pt_GeV"]
    ].mean().round(2)
)