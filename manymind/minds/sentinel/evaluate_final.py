
import argparse
import csv
from pathlib import Path

import numpy as np

from manymind.minds.sentinel.evaluation import (
    HOLDOUT_ANOMALY_MEANS,
    HOLDOUT_SEEDS,
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


def main(
    output_path: Path | None = None,
    config_path=None,
    seeds=HOLDOUT_SEEDS,
    anomaly_means=HOLDOUT_ANOMALY_MEANS,
    **generator_kwargs,
):
    """Score frozen thresholds on holdout data. Does not select a threshold."""
    validate_disjoint_splits()

    if output_path is None:
        destination = default_run_output(
            "evaluate_final",
            "final_threshold_evaluation.csv",
        )
    else:
        destination = resolve_writable_path(output_path)
    model = load_model()
    thresholds = holdout_thresholds(
        config_path=config_path,
        require_calibrated=True,
    )

    X_check, _y_check = generate_labeled_dataset(
        seed=int(seeds[0]),
        anomaly_mean=float(anomaly_means[0]),
        **generator_kwargs,
    )
    assert_native_predict_matches_default_threshold(model, X_check)

    records = []
    pooled_scores = []
    pooled_labels = []

    for anomaly_mean in anomaly_means:
        scores, labels, _means = score_splits(
            model,
            seeds,
            (anomaly_mean,),
            **generator_kwargs,
        )
        pooled_scores.append(scores)
        pooled_labels.append(labels)

        for threshold in thresholds:
            metrics = calculate_metrics(labels, scores, threshold)
            records.append({
                "anomaly_mean": anomaly_mean,
                "split": "holdout",
                "observations": len(labels),
                "actual_anomalies": int(labels.sum()),
                **metrics,
            })

    all_scores = np.concatenate(pooled_scores)
    all_labels = np.concatenate(pooled_labels)

    for threshold in thresholds:
        metrics = calculate_metrics(all_labels, all_scores, threshold)
        records.append({
            "anomaly_mean": "ALL",
            "split": "holdout",
            "observations": len(all_labels),
            "actual_anomalies": int(all_labels.sum()),
            **metrics,
        })

    with destination.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print("Sentinel holdout threshold evaluation")
    print("Split: holdout (threshold was not selected on this data)")
    print(f"Model contamination: {model.contamination}")
    print("Holdout seeds:", list(seeds))
    print("Thresholds:", thresholds)
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
            f"{row['threshold']:10.5f} "
            f"{row['precision']:10.3f} "
            f"{row['recall']:10.3f} "
            f"{row['f1']:10.3f} "
            f"{row['false_positive_rate']:10.3f} "
            f"{row['tp']:6d} {row['fp']:6d} {row['fn']:6d}"
        )

    print(f"\nResults saved to: {destination}")
    print("These synthetic holdout metrics do not establish real-world performance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Sentinel thresholds on holdout data. "
            "Requires a valid threshold.json candidate and writes a "
            "unique run file instead of overwriting historical CSVs."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination CSV path. Defaults to a unique run directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Required calibrated threshold JSON. Defaults to artifacts/sentinel/threshold.json.",
    )
    args = parser.parse_args()
    main(output_path=args.output, config_path=args.config)
