
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from manymind.minds.sentinel.predict import load_model


PROJECT_ROOT = Path("/home/lucky/ManyMind")
DATA_DIR = PROJECT_ROOT / "data" / "sentinel"

ANOMALY_MEANS = [150, 120, 115, 110]
RANDOM_SEEDS = [2026, 2027, 2028, 2029, 2030]


def generate_test_data(anomaly_mean, seed):
    """Generate a reproducible test dataset for one scenario."""
    rng = np.random.default_rng(seed)

    normal = rng.normal(loc=100, scale=5, size=1000)
    anomalies = rng.normal(
        loc=anomaly_mean,
        scale=3,
        size=20,
    )

    X = np.concatenate([normal, anomalies])
    y = np.concatenate([
        np.zeros(len(normal), dtype=int),
        np.ones(len(anomalies), dtype=int),
    ])

    indices = rng.permutation(len(X))
    return X[indices].reshape(-1, 1), y[indices]


def evaluate_robustness():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load the saved model once. Never retrain it here.
    model = load_model()

    print("Sentinel robustness evaluation")
    print(f"Model contamination: {model.contamination}")
    print(f"Scenarios: {len(ANOMALY_MEANS)}")
    print(f"Seeds per scenario: {len(RANDOM_SEEDS)}")

    rows = []

    for anomaly_mean in ANOMALY_MEANS:
        for seed in RANDOM_SEEDS:
            X_test, y_test = generate_test_data(
                anomaly_mean=anomaly_mean,
                seed=seed,
            )

            predictions = model.predict(X_test)
            y_pred = (predictions == -1).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1],
            ).ravel()

            precision, recall, f1, _ = (
                precision_recall_fscore_support(
                    y_test,
                    y_pred,
                    labels=[1],
                    average=None,
                    zero_division=0,
                )
            )

            rows.append({
                "anomaly_mean": anomaly_mean,
                "seed": seed,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "false_positive_rate": float(fp / (fp + tn)),
            })

    runs = pd.DataFrame(rows)

    summary = (
        runs.groupby("anomaly_mean")[
            [
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "false_positives",
                "false_negatives",
            ]
        ]
        .agg(["mean", "std"])
        .round(3)
    )

    runs_path = DATA_DIR / "evaluation_runs.csv"
    summary_path = DATA_DIR / "evaluation_summary.csv"

    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path)

    print("\nAverage results across five seeds per scenario:")

    for anomaly_mean, group in runs.groupby("anomaly_mean"):
        print(f"\nAnomaly mean: {anomaly_mean}")
        print(f"Precision: {group['precision'].mean():.3f}")
        print(f"Recall:    {group['recall'].mean():.3f}")
        print(f"F1-score:  {group['f1'].mean():.3f}")
        print(
            "False positives (mean): "
            f"{group['false_positives'].mean():.1f}"
        )
        print(
            "False negatives (mean): "
            f"{group['false_negatives'].mean():.1f}"
        )
        print(
            "False-positive rate (mean): "
            f"{group['false_positive_rate'].mean():.3f}"
        )

    print(f"\nDetailed results saved to: {runs_path}")
    print(f"Summary saved to: {summary_path}")
    print("\nRobustness evaluation complete.")


if __name__ == "__main__":
    evaluate_robustness()
