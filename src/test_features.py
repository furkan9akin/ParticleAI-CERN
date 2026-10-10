from pathlib import Path
import numpy as np
import uproot
from features import extract_photon_features

# Find the Higgs simulation file
files = list(Path("data/raw").glob("*364352*.root"))

if len(files) != 1:
    raise RuntimeError("Expected exactly one background ROOT file")

# Read only the first 100 events
with uproot.open(files[0]) as root_file:
    tree = root_file["analysis"]

    events = tree.arrays(
        ["photon_pt", "photon_eta", "photon_phi"],
        entry_start=0,
        entry_stop=100,
        library="ak"
    )

# Extract physics features
features = extract_photon_features(events)

print("Number of selected events:",
      len(features["diphoton_mass_GeV"]))

print("\nExtracted features:")
for name in features:
    print(name)

print("\nFirst 5 diphoton masses:")
print(np.round(features["diphoton_mass_GeV"][:5], 2))

# Check that every feature is finite
for name, values in features.items():
    assert np.all(np.isfinite(values)), f"Invalid values in {name}"

print("\nFeature extraction test PASSED")