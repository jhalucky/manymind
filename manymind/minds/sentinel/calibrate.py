
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from manymind.minds.sentinel.predict import load_model


PROJECT_ROOT = Path("/home/lucky/ManyMind")
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "sentinel"

ANOMALY_MEANS = [120, 115, 110]
RANDOM_SEEDS = [3101, 3102, 3103, 3104, 3105]
TARGET_RECALL = 0.90


def generate_validation_data(anomaly_mean, seed):
    """Generate an independent, labeled validation dataset."""
    rng = np.random.default_rng(seed)

    normal = rng.normal(
        loc=100,
        scale=5,
        size=1000,
    )

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


def calculate_metrics(y_true, scores, threshold):
    """Classify using a custom threshold and calculate metrics."""
    # Lower decision_function scores are more anomalous.
    y_pred = (scores < threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0

    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "false_positive_rate": float(fpr),
    }


def calibrate():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()

    all_scores = []
    all_labels = []
    scenario_labels = []

    # Collect validation scores without fitting or modifying the model.
    for anomaly_mean in ANOMALY_MEANS:
        for seed in RANDOM_SEEDS:
            X_val, y_val = generate_validation_data(
                anomaly_mean,
                seed,
            )

            scores = model.decision_function(X_val)

            all_scores.append(scores)
            all_labels.append(y_val)
            scenario_labels.extend([anomaly_mean] * len(y_val))

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    scenario_labels = np.asarray(scenario_labels)

    # Include cutoffs immediately above observed scores so tied scores
    # can be included in the anomaly group.
    thresholds = np.unique(
        np.concatenate([
            scores,
            np.nextafter(scores, np.inf),
        ])
    )

    eligible = []

    for threshold in thresholds:
        metrics = calculate_metrics(labels, scores, threshold)

        if metrics["recall"] >= TARGET_RECALL:
            eligible.append(metrics)

    if not eligible:
        raise RuntimeError(
            f"No threshold achieved recall >= {TARGET_RECALL:.0%}"
        )

    # First minimize false-positive rate. If tied, prefer higher recall,
    # then the lower threshold.
    best = min(
        eligible,
        key=lambda item: (
            item["false_positive_rate"],
            -item["recall"],
            item["threshold"],
        ),
    )

    # Report how the selected threshold behaves in each scenario.
    print("Sentinel threshold calibration")
    print(f"Model contamination: {model.contamination}")
    print(f"Validation observations: {len(labels)}")
    print(f"Target recall: {TARGET_RECALL:.0%}")
    print(f"Selected threshold: {best['threshold']:.8f}")
    print(f"Pooled recall: {best['recall']:.3f}")
    print(f"Pooled precision: {best['precision']:.3f}")
    print(
        "Pooled false-positive rate: "
        f"{best['false_positive_rate']:.3f}"
    )

    print("\nResults by anomaly mean:")

    for anomaly_mean in ANOMALY_MEANS:
        mask = scenario_labels == anomaly_mean

        metrics = calculate_metrics(
            labels[mask],
            scores[mask],
            best["threshold"],
        )

        print(
            f"mean={anomaly_mean} | "
            f"precision={metrics['precision']:.3f} | "
            f"recall={metrics['recall']:.3f} | "
            f"false_positives={metrics['fp']} | "
            f"false_negatives={metrics['fn']}"
        )

    config = {
        "threshold": best["threshold"],
        "target_recall": TARGET_RECALL,
        "selection_metric": "minimum_false_positive_rate",
        "validation_anomaly_means": ANOMALY_MEANS,
        "validation_seeds": RANDOM_SEEDS,
        "validation_metrics": best,
    }

    config_path = ARTIFACT_DIR / "threshold.json"

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print(f"\nThreshold configuration saved to: {config_path}")
    print("The trained model artifact was not modified.")


if __name__ == "__main__":
    calibrate()
