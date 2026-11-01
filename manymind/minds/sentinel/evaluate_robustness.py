
import argparse
from pathlib import Path

import pandas as pd

from manymind.minds.sentinel.evaluation import (
    DEFAULT_THRESHOLD,
    ROBUSTNESS_ANOMALY_MEANS,
    ROBUSTNESS_SEEDS,
    assert_native_predict_matches_default_threshold,
    calculate_metrics,
    iter_labeled_splits,
    new_run_directory,
    resolve_writable_path,
    validate_disjoint_splits,
)
from manymind.minds.sentinel.predict import load_model


def evaluate_robustness(
    data_dir: Path | None = None,
    seeds=ROBUSTNESS_SEEDS,
    anomaly_means=ROBUSTNESS_ANOMALY_MEANS,
    **generator_kwargs,
):
    """Evaluate the default threshold across anomaly-mean scenarios.

    This is a synthetic difficulty sweep, not a real-world robustness proof.
    """
    validate_disjoint_splits()

    if data_dir is None:
        output_dir = new_run_directory("evaluate_robustness")
    else:
        output_dir = Path(data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    runs_path = resolve_writable_path(output_dir / "evaluation_runs.csv")
    summary_path = resolve_writable_path(output_dir / "evaluation_summary.csv")

    model = load_model()

    print("Sentinel robustness evaluation")
    print("Split: robustness (default threshold 0.0)")
    print(f"Model contamination: {model.contamination}")
    print(f"Scenarios: {len(anomaly_means)}")
    print(f"Seeds per scenario: {len(seeds)}")

    rows = []
    checked = False

    for anomaly_mean, seed, X_test, y_test in iter_labeled_splits(
        seeds,
        anomaly_means,
        **generator_kwargs,
    ):
        if not checked:
            assert_native_predict_matches_default_threshold(model, X_test)
            checked = True

        scores = model.score(X_test)
        metrics = calculate_metrics(y_test, scores, DEFAULT_THRESHOLD)

        rows.append({
            "anomaly_mean": anomaly_mean,
            "seed": seed,
            "true_positives": metrics["tp"],
            "false_positives": metrics["fp"],
            "false_negatives": metrics["fn"],
            "true_negatives": metrics["tn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "false_positive_rate": metrics["false_positive_rate"],
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

    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path)

    print("\nAverage results across seeds per scenario:")

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
    print("These synthetic metrics do not establish real-world performance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Sentinel robustness at the default 0.0 threshold. "
            "Writes unique run files instead of overwriting historical CSVs."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for robustness CSVs. Defaults to a unique run directory.",
    )
    evaluate_robustness(data_dir=parser.parse_args().output_dir)
