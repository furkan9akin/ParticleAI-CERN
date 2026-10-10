from pathlib import Path
import uproot
import awkward as ak
import matplotlib.pyplot as plt

# Find the downloaded CERN ROOT file
root_files = list(Path("data/raw").glob("*.root"))

if len(root_files) != 1:
    raise RuntimeError("Expected exactly one ROOT file in data/raw")

# Read photon transverse momentum
with uproot.open(root_files[0]) as root_file:
    tree = root_file["analysis"]
    events = tree.arrays(["photon_pt"], library="ak")

# Convert variable-length photon arrays to a flat array
photon_pt = ak.to_numpy(ak.flatten(events["photon_pt"]))

# Create histogram
plt.figure(figsize=(8, 5))
plt.hist(photon_pt, bins=20, edgecolor="black")

plt.title("Photon Transverse Momentum Distribution")
plt.xlabel("Photon transverse momentum (GeV)")
plt.ylabel("Number of photons")

plt.tight_layout()

# Save the figure
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/photon_pt_distribution.png", dpi=150)

plt.show()

print("\nPhoton momentum summary")
print("-----------------------")
print("Number of photons:", len(photon_pt))
print("Minimum pT:", round(photon_pt.min(), 2))
print("Maximum pT:", round(photon_pt.max(), 2))
print("Mean pT:", round(photon_pt.mean(), 2))
print("Median pT:", round(float(__import__("numpy").median(photon_pt)), 2))