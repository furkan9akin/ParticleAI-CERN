from pathlib import Path
import uproot
import awkward as ak

# Find the ROOT file
root_files = list(Path("data/raw").glob("*.root"))

if len(root_files) != 1:
    raise RuntimeError("Expected exactly one ROOT file")

with uproot.open(root_files[0]) as root_file:
    tree = root_file["analysis"]
    photon_pt = tree["photon_pt"].array(library="ak")

# Sort photons by transverse momentum (highest first)
sorted_indices = ak.argsort(photon_pt, axis=1, ascending=False)
sorted_pt = photon_pt[sorted_indices]

print("First 5 events - leading and subleading photons:")

for i in range(5):
    print(f"\nEvent {i + 1}")
    print("Leading photon pT:", round(float(sorted_pt[i][0]), 2))
    print("Subleading photon pT:", round(float(sorted_pt[i][1]), 2))