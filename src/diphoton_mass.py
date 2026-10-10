from pathlib import Path
import numpy as np
import awkward as ak
import uproot

# Locate the ROOT file
root_files = list(Path("data/raw").glob("*.root"))
# Select the ggH125 Higgs simulation
root_files = list(Path("data/raw").glob("*.root"))

higgs_files = [
    path for path in root_files
    if "343981" in path.name
]

if len(higgs_files) != 1:
    raise RuntimeError("Expected exactly one ggH125 ROOT file")

file_path = higgs_files[0]

print("Selected sample:", file_path.name)
with uproot.open(file_path) as root_file:
    tree = root_file["analysis"]
    photons = tree.arrays(
        ["photon_pt", "photon_eta", "photon_phi"],
        library="ak"
    )

# Sort photons by transverse momentum
indices = ak.argsort(
    photons["photon_pt"],
    axis=1,
    ascending=False
)

# Use float64 for more accurate physics calculations
pt = ak.values_astype(
    photons["photon_pt"][indices], np.float64
)

eta = ak.values_astype(
    photons["photon_eta"][indices], np.float64
)

phi = ak.values_astype(
    photons["photon_phi"][indices], np.float64
)

# Keep events with at least two photons
valid = ak.num(pt) >= 2

pt = pt[valid]
eta = eta[valid]
phi = phi[valid]

# Select the two leading photons
pt1 = pt[:, 0]
pt2 = pt[:, 1]

eta1 = eta[:, 0]
eta2 = eta[:, 1]

phi1 = phi[:, 0]
phi2 = phi[:, 1]

# Calculate angular differences
delta_eta = eta1 - eta2
delta_phi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi

# Diphoton invariant mass
mass_squared = (
    2 * pt1 * pt2 *
    (np.cosh(delta_eta) - np.cos(delta_phi))
)

mass = np.sqrt(np.maximum(mass_squared, 0))

print("Number of selected events:", len(mass))
print("\nFirst 5 diphoton invariant masses:")

for i, value in enumerate(mass[:5]):
    print(f"Event {i + 1}: {float(value):.2f} GeV")

import vector

# Construct photon four-vectors
photon1 = vector.array({
    "pt": ak.to_numpy(pt1),
    "eta": ak.to_numpy(eta1),
    "phi": ak.to_numpy(phi1),
    "mass": np.zeros(len(pt1))
})

photon2 = vector.array({
    "pt": ak.to_numpy(pt2),
    "eta": ak.to_numpy(eta2),
    "phi": ak.to_numpy(phi2),
    "mass": np.zeros(len(pt2))
})

# Independent mass calculation
mass_vector = (photon1 + photon2).mass

# Compare both methods
difference = np.abs(ak.to_numpy(mass) - mass_vector)

print("\nPhysics validation:")
print("Maximum mass difference:", difference.max(), "GeV")

# Investigate numerical differences
mass_formula = ak.to_numpy(mass)

print("\nNumerical precision check:")
print("Formula dtype:", mass_formula.dtype)
print("Vector dtype:", mass_vector.dtype)

# Find the event with the largest difference
max_index = np.argmax(difference)

print("Event index:", max_index)
print("Formula mass:", mass_formula[max_index])
print("Vector mass:", mass_vector[max_index])
print("Absolute difference:", difference[max_index])

# Compare relative difference for nonzero masses
relative_diff = difference / np.maximum(
    np.abs(mass_vector), 1e-12
)

print("Maximum relative difference:", relative_diff.max())


# Final validation of the two calculation methods
assert np.allclose(
    ak.to_numpy(mass),
    mass_vector,
    rtol=1e-8,
    atol=1e-8
), "Diphoton mass calculations do not agree!"

print("\nMass calculation validation PASSED")

import matplotlib.pyplot as plt

# Create diphoton invariant mass histogram
mass_values = ak.to_numpy(mass)

plt.figure(figsize=(8, 5))

# Focus on the Higgs mass region for visualization
plt.hist(
    mass_values,
    bins=50,
    range=(100, 150),
    edgecolor="black"
)

plt.xlim(100, 150)

plt.title("Diphoton Invariant Mass - ggH125 Signal MC (100-150 GeV)")
plt.xlabel("Diphoton invariant mass (GeV)")
plt.ylabel("Number of events")

plt.tight_layout()

# Save figure
Path("figures").mkdir(exist_ok=True)

plt.savefig(
    "figures/higgs_mass_zoom.png",
    dpi=150
)

plt.show()