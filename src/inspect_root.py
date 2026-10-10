from pathlib import Path
import uproot

# CERN ROOT dosyasının bulunduğu klasör
data_folder = Path("data/raw")

# Klasördeki ROOT dosyalarını bul
root_files = list(data_folder.glob("*.root"))

if not root_files:
    raise FileNotFoundError("No ROOT files found in data/raw")

# Select the Higgs simulation sample explicitly
higgs_files = [
    path for path in root_files
    if "343981" in path.name
]

if len(higgs_files) != 1:
    raise RuntimeError("Expected exactly one ggH125 ROOT file")

file_path = higgs_files[0]

print("File:", file_path.name)

with uproot.open(file_path) as root_file:
    print("\nROOT file contents:")
    print(root_file.keys())

    tree = root_file["analysis"]

    print("\nNumber of events:")
    print(tree.num_entries)

    print("\nAvailable branches:")
    for branch in tree.keys():
        print(branch)

    print("\nFirst 5 events:")

    events = tree.arrays(
        ["photon_n", "photon_pt", "photon_eta", "photon_phi"],
        entry_start=0,
        entry_stop=5,
        library="ak"
    )

    for i in range(len(events)):
        print(f"\nEvent {i + 1}")
        print("Photon count:", events["photon_n"][i])
        print("Photon pt:", events["photon_pt"][i].to_list())
        import numpy as np

    photon_counts = tree["photon_n"].array(library="np")

    values, counts = np.unique(
        photon_counts,
        return_counts=True
    )

    print("\nPhoton multiplicity:")
    for value, count in zip(values, counts):
        print(f"{value} photons: {count} events")