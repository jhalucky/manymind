
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from manymind.minds.sentinel.predict import load_model


PROJECT_ROOT = Path("/home/lucky/ManyMind")
DATA_DIR = PROJECT_ROOT / "data" / "sentinel"


def generate_evaluation_data():
    """Generate fresh data independent of training and tuning."""
    rng = np.random.default_rng(2026)

    normal = rng.normal(
        loc=100,
        scale=5,
        size=1000,
    )

    anomalies = rng.normal(
        loc=120,
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


def evaluate():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load the existing artifact; do not train again.
    model = load_model()

    X_test, y_test = generate_evaluation_data()

    # Save the exact evaluation data for reproducibility.
    import pandas as pd

    pd.DataFrame({
        "value": X_test.flatten(),
        "is_anomaly": y_test,
    }).to_csv(DATA_DIR / "final_test.csv", index=False)

    predictions = model.predict(X_test)
    y_pred = (predictions == -1).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

    print("Sentinel final evaluation")
    print(f"Model contamination: {model.contamination}")
    print(f"Test observations: {len(X_test)}")
    print(f"Actual anomalies: {int(y_test.sum())}")
    print(f"True positives: {tp}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True negatives: {tn}")

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

    print(f"Evaluation data saved to: {DATA_DIR / 'final_test.csv'}")


if __name__ == "__main__":
    evaluate()
