
import argparse
import json
from pathlib import Path

import numpy as np

from manymind.minds.sentinel.evaluation import (
    CALIBRATION_ANOMALY_MEANS,
    CALIBRATION_SEEDS,
    TARGET_RECALL,
    assert_native_predict_matches_default_threshold,
    calculate_metrics,
    default_run_output,
    resolve_writable_path,
    score_splits,
    validate_disjoint_splits,
)
from manymind.minds.sentinel.predict import load_model
from manymind.minds.sentinel.synthetic import generate_labeled_dataset


def calibrate(
    output_path: Path | None = None,
    seeds=CALIBRATION_SEEDS,
    anomaly_means=CALIBRATION_ANOMALY_MEANS,
    target_recall: float = TARGET_RECALL,
    **generator_kwargs,
):
    """Select a threshold on calibration data only. Does not use holdout data."""
    validate_disjoint_splits()

    if output_path is None:
        config_path = default_run_output("calibrate", "threshold.json")
    else:
        config_path = resolve_writable_path(output_path)

    model = load_model()

    scores, labels, scenario_labels = score_splits(
        model,
        seeds,
        anomaly_means,
        **generator_kwargs,
    )

    X_check, _y_check = generate_labeled_dataset(
        seed=int(seeds[0]),
        anomaly_mean=float(anomaly_means[0]),
        **generator_kwargs,
    )
    assert_native_predict_matches_default_threshold(model, X_check)

    n_observations = int(len(labels))
    n_anomalies = int(labels.sum())
    prevalence = n_anomalies / n_observations if n_observations else 0.0

    thresholds = np.unique(
        np.concatenate([
            scores,
            np.nextafter(scores, np.inf),
        ])
    )

    eligible = []
    for threshold in thresholds:
        metrics = calculate_metrics(labels, scores, threshold)
        if metrics["recall"] >= target_recall:
            eligible.append(metrics)

    if not eligible:
        raise RuntimeError(
            f"No threshold achieved recall >= {target_recall:.0%}"
        )

    best = min(
        eligible,
        key=lambda item: (
            item["false_positive_rate"],
            -item["recall"],
            item["threshold"],
        ),
    )

    print("Sentinel threshold calibration")
    print("Split: calibration (not holdout)")
    print(f"Model contamination: {model.contamination}")
    print(f"Calibration observations: {n_observations}")
    print(f"Calibration anomalies: {n_anomalies}")
    print(f"Anomaly prevalence: {prevalence:.4f}")
    print(f"Target recall: {target_recall:.0%}")
    print(f"Selected threshold: {best['threshold']:.8f}")
    print(f"True positives: {best['tp']}")
    print(f"False positives: {best['fp']}")
    print(f"False negatives: {best['fn']}")
    print(f"True negatives: {best['tn']}")
    print(f"Pooled recall: {best['recall']:.3f}")
    print(f"Pooled precision: {best['precision']:.3f}")
    print(
        "Pooled false-positive rate: "
        f"{best['false_positive_rate']:.3f}"
    )

    print("\nResults by anomaly mean:")

    for anomaly_mean in anomaly_means:
        mask = scenario_labels == float(anomaly_mean)
        metrics = calculate_metrics(
            labels[mask],
            scores[mask],
            best["threshold"],
        )
        print(
            f"mean={anomaly_mean} | "
            f"tp={metrics['tp']} fp={metrics['fp']} "
            f"fn={metrics['fn']} tn={metrics['tn']} | "
            f"precision={metrics['precision']:.3f} | "
            f"recall={metrics['recall']:.3f} | "
            f"false_positive_rate={metrics['false_positive_rate']:.3f}"
        )

    config = {
        "threshold": best["threshold"],
        "target_recall": target_recall,
        "selection_metric": "minimum_false_positive_rate",
        "split": "calibration",
        "validation_anomaly_means": list(anomaly_means),
        "validation_seeds": list(seeds),
        "n_observations": n_observations,
        "n_anomalies": n_anomalies,
        "anomaly_prevalence": prevalence,
        "validation_metrics": best,
    }

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print(f"\nThreshold configuration saved to: {config_path}")
    print("The trained model artifact was not modified.")
    print("Holdout data were not used to select this threshold.")

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a Sentinel threshold on calibration data. "
            "Writes a new run file and does not overwrite "
            "artifacts/sentinel/threshold.json unless that path is "
            "passed explicitly and does not already exist."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination JSON path. Defaults to a unique run directory.",
    )
    calibrate(output_path=parser.parse_args().output)
