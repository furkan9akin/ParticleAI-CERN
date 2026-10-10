import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# Load the previously saved model and metadata
model_path = Path("models/gradient_boosting_v1.joblib")
metadata_path = Path("models/gradient_boosting_v1_metadata.json")

model = joblib.load(model_path)

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)

# Load the untouched test dataset
test = pd.read_csv("data/processed/test.csv")

FEATURES = metadata["features"]

X_test = test[FEATURES]
y_test = test["label"]

# Use the previously fixed classification threshold
threshold = metadata["classification_threshold"]

scores = model.predict_proba(X_test)[:, 1]
predictions = (scores >= threshold).astype(int)

print("FINAL TEST - Gradient Boosting")
print("------------------------------")

print("Test events:", len(test))
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions), 4))
print("Recall:", round(recall_score(y_test, predictions), 4))
print("F1:", round(f1_score(y_test, predictions), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, scores), 4))
print("PR-AUC:", round(average_precision_score(y_test, scores), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nFinal test completed")

# Save final test results for documentation
results = {
    "model": "Gradient Boosting",
    "dataset": "ATLAS 2025 Monte Carlo",
    "test_events": len(test),
    "accuracy": accuracy_score(y_test, predictions),
    "precision": precision_score(y_test, predictions),
    "recall": recall_score(y_test, predictions),
    "f1": f1_score(y_test, predictions),
    "roc_auc": roc_auc_score(y_test, scores),
    "pr_auc": average_precision_score(y_test, scores),
    "confusion_matrix": confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).tolist(),
    "limitations": (
        "Balanced unweighted Monte Carlo samples; "
        "not a real collision Higgs discovery claim."
    )
}

output = Path("report")
output.mkdir(exist_ok=True)

with open(
    output / "final_test_results.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=2)

print("\nFinal results saved to report/final_test_results.json")