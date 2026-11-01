
import argparse
import csv
from pathlib import Path

from manymind.minds.sentinel.evaluation import (
    CALIBRATION_ANOMALY_MEANS,
    CALIBRATION_SEEDS,
    COMPARISON_THRESHOLDS,
    assert_native_predict_matches_default_threshold,
    calculate_metrics,
    default_run_output,
    holdout_thresholds,
    resolve_writable_path,
    score_splits,
    validate_disjoint_splits,
)
from manymind.minds.sentinel.predict import load_model
from manymind.minds.sentinel.synthetic import generate_labeled_dataset


def comparison_thresholds(config_path=None):
    """Explore default, historical, and saved-candidate cutoffs on calibration data."""
    values = holdout_thresholds(
        extra=COMPARISON_THRESHOLDS[1:],
        config_path=config_path,
        require_calibrated=False,
    )
    for value in COMPARISON_THRESHOLDS:
        if not any(abs(value - existing) < 1e-12 for existing in values):
            values.append(float(value))
    return values


def main(
    output_path: Path | None = None,
    config_path=None,
    seeds=CALIBRATION_SEEDS,
    anomaly_means=CALIBRATION_ANOMALY_MEANS,
    **generator_kwargs,
):
    """Compare thresholds on calibration data. Does not use the holdout split."""
    validate_disjoint_splits()

    if output_path is None:
        destination = default_run_output(
            "compare_thresholds",
            "threshold_comparison.csv",
        )
    else:
        destination = resolve_writable_path(output_path)
    model = load_model()
    thresholds = comparison_thresholds(config_path=config_path)

    X_check, _y_check = generate_labeled_dataset(
        seed=int(seeds[0]),
        anomaly_mean=float(anomaly_means[0]),
        **generator_kwargs,
    )
    assert_native_predict_matches_default_threshold(model, X_check)

    scores, labels, means = score_splits(
        model,
        seeds,
        anomaly_means,
        **generator_kwargs,
    )

    results = []
    for threshold in thresholds:
        row = calculate_metrics(labels, scores, threshold)
        csv_row = {
            "threshold": row["threshold"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
            "false_positive_rate": row["false_positive_rate"],
            "true_positives": row["tp"],
            "false_positives": row["fp"],
            "false_negatives": row["fn"],
            "true_negatives": row["tn"],
        }
        for mean in anomaly_means:
            mask = means == float(mean)
            scenario = calculate_metrics(labels[mask], scores[mask], threshold)
            csv_row[f"recall_mean_{int(mean)}"] = scenario["recall"]
            csv_row[f"fpr_mean_{int(mean)}"] = scenario["false_positive_rate"]
        results.append(csv_row)

    with destination.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("Sentinel threshold comparison")
    print("Split: calibration (not holdout)")
    print(f"Model contamination: {model.contamination}")
    print(f"Calibration observations: {len(labels)}")
    print(f"Actual anomalies: {int(labels.sum())}")
    recall_keys = [f"recall_mean_{int(mean)}" for mean in anomaly_means]

    print("\nThreshold comparison")
    print("-" * 100)

    header = (
        f"{'Threshold':>12} {'Precision':>10} {'Recall':>10} "
        f"{'F1':>10} {'FPR':>10}"
    )
    for mean in anomaly_means:
        header += f" {'Recall ' + str(int(mean)):>12}"
    print(header)

    for row in results:
        line = (
            f"{row['threshold']:12.5f} "
            f"{row['precision']:10.3f} "
            f"{row['recall']:10.3f} "
            f"{row['f1']:10.3f} "
            f"{row['false_positive_rate']:10.3f}"
        )
        for key in recall_keys:
            line += f" {row[key]:12.3f}"
        print(line)

    print(f"\nDetailed results saved to: {destination}")
    print("Holdout data were not used.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare Sentinel thresholds on calibration data. "
            "Writes a unique run file instead of overwriting "
            "data/sentinel/threshold_comparison.csv."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination CSV path. Defaults to a unique run directory.",
    )
    main(output_path=parser.parse_args().output)
