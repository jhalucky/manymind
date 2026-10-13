
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from manymind.minds.sentinel.predict import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = PROJECT_ROOT / "data" / "sentinel" / "final_threshold_evaluation.csv"

# Fresh seeds not used in the earlier calibration/comparison.
RANDOM_SEEDS = [9101, 9102, 9103, 9104, 9105]

ANOMALY_MEANS = [120, 115, 110]
THRESHOLDS = [0.0, 0.05]


def generate_test_data(anomaly_mean, seed):
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
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def main():
    model = load_model()
    records = []

    for anomaly_mean in ANOMALY_MEANS:
        scenario_scores = []
        scenario_labels = []

        for seed in RANDOM_SEEDS:
            X, y = generate_test_data(anomaly_mean, seed)
            scores = model.decision_function(X)

            scenario_scores.append(scores)
            scenario_labels.append(y)

        scores = np.concatenate(scenario_scores)
        labels = np.concatenate(scenario_labels)

        for threshold in THRESHOLDS:
            metrics = calculate_metrics(labels, scores, threshold)
            records.append({
                "anomaly_mean": anomaly_mean,
                "observations": len(labels),
                "actual_anomalies": int(labels.sum()),
                **metrics,
            })

    # Overall metrics across all three scenarios.
    overall_records = []

    for threshold in THRESHOLDS:
        all_scores = []
        all_labels = []

        for anomaly_mean in ANOMALY_MEANS:
            for seed in RANDOM_SEEDS:
                X, y = generate_test_data(anomaly_mean, seed)
                all_scores.append(model.decision_function(X))
                all_labels.append(y)

        metrics = calculate_metrics(
            np.concatenate(all_labels),
            np.concatenate(all_scores),
            threshold,
        )

        overall_records.append({
            "anomaly_mean": "ALL",
            "observations": sum(len(y) for y in all_labels),
            "actual_anomalies": sum(int(y.sum()) for y in all_labels),
            **metrics,
        })

    records.extend(overall_records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"Model contamination: {model.contamination}")
    print("Fresh evaluation seeds:", RANDOM_SEEDS)
    print("Thresholds:", THRESHOLDS)
    print("\nFinal threshold evaluation")
    print("-" * 95)
    print(
        f"{'Scenario':>10} {'Threshold':>10} {'Precision':>10} "
        f"{'Recall':>10} {'F1':>10} {'FPR':>10} "
        f"{'TP':>6} {'FP':>6} {'FN':>6}"
    )

    for row in records:
        print(
            f"{str(row['anomaly_mean']):>10} "
            f"{row['threshold']:10.3f} "
            f"{row['precision']:10.3f} "
            f"{row['recall']:10.3f} "
            f"{row['f1']:10.3f} "
            f"{row['false_positive_rate']:10.3f} "
            f"{row['tp']:6d} {row['fp']:6d} {row['fn']:6d}"
        )

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
