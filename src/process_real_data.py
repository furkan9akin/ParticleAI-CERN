from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from features import extract_photon_features


# Find real ATLAS data
files = list(Path("data/raw").glob("*data15_periodD*.root"))

if len(files) != 1:
    raise RuntimeError("Expected exactly one real data ROOT file")

file_path = files[0]

print("Processing:", file_path.name)

# Read the same photon variables used for MC
with uproot.open(file_path) as root_file:
    tree = root_file["analysis"]

    events = tree.arrays(
        ["photon_pt", "photon_eta", "photon_phi"],
        library="ak"
    )

# Reuse exactly the same feature extraction function
features = extract_photon_features(events)

real_df = pd.DataFrame(features)

# Validate numerical results
numeric_values = real_df.to_numpy()

assert np.all(np.isfinite(numeric_values)), \
    "Invalid numerical values detected"

# Do not create signal/background labels for real data
real_df["sample_name"] = "data15_periodD"

# Save processed data locally
output = Path("data/processed")
output.mkdir(parents=True, exist_ok=True)

output_file = output / "real_data_processed.csv"
real_df.to_csv(output_file, index=False)

print("\nReal data processing completed")
print("Processed events:", len(real_df))
print("Missing values:", real_df.isna().sum().sum())
print("Features:", len(features))

print("\nDiphoton mass summary (GeV):")
print(
    real_df["diphoton_mass_GeV"]
    .describe()
    .round(2)
)

print("\nSaved to:", output_file)