from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

df = pd.read_csv("data/processed/ml_dataset.csv")

# Split 70% training and 30% remaining
train, remaining = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=df["label"]
)

# Split remaining equally into validation and test
validation, test = train_test_split(
    remaining,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=remaining["label"]
)

output = Path("data/processed")
output.mkdir(parents=True, exist_ok=True)

train.to_csv(output / "train.csv", index=False)
validation.to_csv(output / "validation.csv", index=False)
test.to_csv(output / "test.csv", index=False)

for name, dataset in [
    ("Training", train),
    ("Validation", validation),
    ("Test", test)
]:
    print(f"\n{name}: {len(dataset)} events")
    print(dataset["label"].value_counts().sort_index())

print("\nDataset split completed successfully")