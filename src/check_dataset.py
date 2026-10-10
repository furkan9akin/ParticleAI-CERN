import numpy as np
import pandas as pd

# Load the prepared dataset
df = pd.read_csv("data/processed/ml_dataset.csv")

print("Dataset shape:", df.shape)

print("\nMissing values:")
print(df.isna().sum())

# Check infinite values in numerical columns
numeric = df.select_dtypes(include="number")

print("\nInfinite values:")
print(np.isinf(numeric.to_numpy()).sum())

print("\nClass counts:")
print(df["label"].value_counts())

# Inspect diphoton mass distribution for each class
print("\nDiphoton mass summary by class:")
print(
    df.groupby("label")["diphoton_mass_GeV"]
    .agg(["min", "median", "mean", "max"])
    .round(2)
)

# Compare populations in the 120-130 GeV region
print("\nEvents in 120-130 GeV:")
for label in [0, 1]:
    subset = df[df["label"] == label]
    count = subset["diphoton_mass_GeV"].between(
        120, 130
    ).sum()

    print(
        f"Class {label}: {count} / {len(subset)} "
        f"({100 * count / len(subset):.1f}%)"
    )