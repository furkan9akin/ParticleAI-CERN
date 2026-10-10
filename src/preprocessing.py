from pathlib import Path
import numpy as np
import pandas as pd
import uproot

from features import extract_photon_features


RANDOM_SEED = 42
EVENTS_PER_CLASS = 10000

DATA_FOLDER = Path("data/raw")
OUTPUT_FOLDER = Path("data/processed")


def load_sample(sample_id, label):
    """Load a reproducible random sample of MC events."""

    files = list(DATA_FOLDER.glob(f"*{sample_id}*.root"))

    if len(files) != 1:
        raise RuntimeError(
            f"Expected one ROOT file for sample {sample_id}"
        )

    with uproot.open(files[0]) as root_file:
        tree = root_file["analysis"]
        total_events = tree.num_entries

        # Select random event positions reproducibly
        rng = np.random.default_rng(RANDOM_SEED)

        selected_indices = np.sort(
            rng.choice(
                total_events,
                size=min(EVENTS_PER_CLASS, total_events),
                replace=False
            )
        )

        # Read only required branches in manageable chunks
        selected_parts = []

        start = 0
        chunk_size = 50000

        while start < total_events:
            stop = min(start + chunk_size, total_events)

            positions = selected_indices[
                (selected_indices >= start) &
                (selected_indices < stop)
            ]

            if len(positions) > 0:
                events = tree.arrays(
                    ["photon_pt", "photon_eta", "photon_phi"],
                    entry_start=start,
                    entry_stop=stop,
                    library="ak"
                )

                local_positions = positions - start
                events = events[local_positions]

                features = extract_photon_features(events)
                selected_parts.append(pd.DataFrame(features))

            start = stop

    if not selected_parts:
        raise RuntimeError("No events selected")

    result = pd.concat(selected_parts, ignore_index=True)

    result["label"] = label
    result["sample_name"] = str(sample_id)

    return result


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print("Loading Higgs signal sample...")
    signal = load_sample("343981", 1)

    print("Loading diphoton background sample...")
    background = load_sample("364352", 0)

    dataset = pd.concat(
        [signal, background],
        ignore_index=True
    )

    # Shuffle rows reproducibly
    dataset = dataset.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    output_path = OUTPUT_FOLDER / "ml_dataset.csv"
    dataset.to_csv(output_path, index=False)

    print("\nDataset created successfully")
    print("Total events:", len(dataset))
    print("\nClass counts:")
    print(dataset["label"].value_counts())
    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()