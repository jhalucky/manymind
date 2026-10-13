
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from manymind.minds.sentinel.predict import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = PROJECT_ROOT / "data" / "sentinel" / "threshold_comparison.csv"

ANOMALY_MEANS = [120, 115, 110]
RANDOM_SEEDS = [3101, 3102, 3103, 3104, 3105]

THRESHOLDS = [
    0.0,          # Existing IsolationForest decision boundary
    0.02,
    0.05,
    0.08,
    0.10,
    0.13176012,   # Previously calibrated candidate
]


def generate_validation_data(anomaly_mean, seed):
    rng = np.random.default_rng(seed)

    normal = rng.normal(loc=100, scale=5, size=1000)
    anomalies = rng.normal(loc=anomaly_mean, scale=3, size=20)

    X = np.concatenate([normal, anomalies])
    y = np.concatenate([
        np.zeros(len(normal), dtype=int),
        np.ones(len(anomalies), dtype=int),
    ])

    indices = rng.permutation(len(X))
    return X[indices].reshape(-1, 1), y[indices]


def calculate_metrics(y_true, scores, threshold):
    # Lower scores indicate more anomalous observations.
    y_pred = (scores < threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    fpr = fp / (fp + tn) if fp + tn else 0.0

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def main():
    model = load_model()

    all_scores = []
    all_labels = []
    all_means = []

    for anomaly_mean in ANOMALY_MEANS:
        for seed in RANDOM_SEEDS:
            X, y = generate_validation_data(anomaly_mean, seed)

            scores = model.decision_function(X)

            all_scores.append(scores)
            all_labels.append(y)
            all_means.extend([anomaly_mean] * len(y))

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    means = np.asarray(all_means)

    results = []

    for threshold in THRESHOLDS:
        row = calculate_metrics(labels, scores, threshold)

        # Recall and false-positive rate for each anomaly scenario.
        for mean in ANOMALY_MEANS:
            mask = means == mean
            scenario = calculate_metrics(
                labels[mask], scores[mask], threshold
            )

            row[f"recall_mean_{mean}"] = scenario["recall"]
            row[f"fpr_mean_{mean}"] = scenario["false_positive_rate"]

        results.append(row)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Model contamination: {model.contamination}")
    print(f"Validation observations: {len(labels)}")
    print(f"Actual anomalies: {int(labels.sum())}")
    print("\nThreshold comparison")
    print("-" * 100)

    print(
        f"{'Threshold':>12} {'Precision':>10} {'Recall':>10} "
        f"{'F1':>10} {'FPR':>10} "
        f"{'Recall 120':>12} {'Recall 115':>12} "
        f"{'Recall 110':>12}"
    )

    for row in results:
        print(
            f"{row['threshold']:12.5f} "
            f"{row['precision']:10.3f} "
            f"{row['recall']:10.3f} "
            f"{row['f1']:10.3f} "
            f"{row['false_positive_rate']:10.3f} "
            f"{row['recall_mean_120']:12.3f} "
            f"{row['recall_mean_115']:12.3f} "
            f"{row['recall_mean_110']:12.3f}"
        )

    print(f"\nDetailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
