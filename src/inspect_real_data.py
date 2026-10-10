from pathlib import Path
import uproot

# Find the real ATLAS collision data file
files = list(
    Path("data/raw").glob("*data15_periodD*.root")
)

if len(files) != 1:
    raise RuntimeError("Expected exactly one real data file")

file_path = files[0]

print("Real data file:", file_path.name)
print("File size (MB):", round(file_path.stat().st_size / 1_000_000, 2))

with uproot.open(file_path) as root_file:
    print("\nROOT objects:", root_file.keys())

    tree = root_file["analysis"]

    print("Number of events:", tree.num_entries)

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

    # Read only the first five events
    events = tree.arrays(
        ["photon_n", "photon_pt"],
        entry_start=0,
        entry_stop=5,
        library="ak"
    )

    print("\nFirst 5 real collision events:")

    for i in range(len(events)):
        print(f"\nEvent {i + 1}")
        print("Photon count:", events["photon_n"][i])
        print("Photon pT:", events["photon_pt"][i].to_list())