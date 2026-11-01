
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from manymind.minds.sentinel.evaluation import (
    DEFAULT_THRESHOLD,
    HOLDOUT_SEEDS,
    assert_native_predict_matches_default_threshold,
    calculate_metrics,
    labels_from_scores,
    new_run_directory,
    resolve_writable_path,
    validate_disjoint_splits,
)
from manymind.minds.sentinel.predict import load_model
from manymind.minds.sentinel.synthetic import generate_labeled_dataset


def generate_evaluation_data(seed: int = HOLDOUT_SEEDS[0], anomaly_mean: float = 120):
    """Generate a labeled holdout snapshot. Not used for threshold selection."""
    return generate_labeled_dataset(seed=seed, anomaly_mean=anomaly_mean)


def evaluate(data_dir: Path | None = None):
    validate_disjoint_splits()

    if data_dir is None:
        output_dir = new_run_directory("evaluate")
    else:
        output_dir = Path(data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    destination = resolve_writable_path(output_dir / "final_test.csv")

    model = load_model()
    X_test, y_test = generate_evaluation_data()
    assert_native_predict_matches_default_threshold(model, X_test)

    pd.DataFrame({
        "value": X_test.flatten(),
        "is_anomaly": y_test,
    }).to_csv(destination, index=False)

    scores = model.score(X_test)
    y_pred = labels_from_scores(scores, DEFAULT_THRESHOLD)
    metrics = calculate_metrics(y_test, scores, DEFAULT_THRESHOLD)

    print("Sentinel holdout evaluation")
    print("Split: holdout (default threshold 0.0)")
    print(f"Model contamination: {model.contamination}")
    print(f"Test observations: {len(X_test)}")
    print(f"Actual anomalies: {int(y_test.sum())}")
    print(f"True positives: {metrics['tp']}")
    print(f"False positives: {metrics['fp']}")
    print(f"False negatives: {metrics['fn']}")
    print(f"True negatives: {metrics['tn']}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["normal", "anomaly"],
        zero_division=0,
    ))

    print(f"Evaluation data saved to: {destination}")
    print("These synthetic metrics do not establish real-world performance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Sentinel at the default 0.0 threshold. "
            "Writes a unique run file instead of overwriting "
            "data/sentinel/final_test.csv."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for final_test.csv. Defaults to a unique run directory.",
    )
    evaluate(data_dir=parser.parse_args().output_dir)
