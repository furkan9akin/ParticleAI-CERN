from pathlib import Path
import uproot

# Find the background simulation file
root_files = list(Path("data/raw").glob("*364352*.root"))

if len(root_files) != 1:
    raise RuntimeError(
        f"Expected one background ROOT file, found {len(root_files)}"
    )

file_path = root_files[0]

print("Background file:", file_path.name)
print("File size (MB):", round(file_path.stat().st_size / 1_000_000, 2))

# Open the CERN ROOT file
with uproot.open(file_path) as root_file:
    print("\nROOT objects:", root_file.keys())

    tree = root_file["analysis"]

    print("Number of events:", tree.num_entries)

    # Check whether the required photon variables exist
    required_branches = [
        "photon_n",
        "photon_pt",
        "photon_eta",
        "photon_phi",
        "photon_e"
    ]

    print("\nPhoton branch check:")

    for branch in required_branches:
        if branch in tree.keys():
            print(branch, ": OK")
        else:
            print(branch, ": MISSING")
        # Inspect only the first five background events
    events = tree.arrays(
        ["photon_n", "photon_pt", "photon_eta", "photon_phi"],
        entry_start=0,
        entry_stop=5,
        library="ak"
    )

    print("\nFirst 5 background events:")

    for i in range(len(events)):
        print(f"\nEvent {i + 1}")
        print("Photon count:", events["photon_n"][i])
        print("Photon pT:", events["photon_pt"][i].to_list())