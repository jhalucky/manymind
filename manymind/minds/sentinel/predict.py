
import argparse
import math
from pathlib import Path

import pandas as pd

from manymind.minds.sentinel.data import FEATURES_NAMES
from manymind.minds.sentinel.evaluation import load_calibrated_threshold
from manymind.minds.sentinel.model import DEFAULT_THRESHOLD, Sentinel


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "sentinel" / "model.joblib"


def load_model():
    """Load the persisted Sentinel model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Sentinel model not found: {MODEL_PATH}\n"
            "Train a model and save the artifact first."
        )

    return Sentinel().load(MODEL_PATH)


def predict_value(model, value, threshold=DEFAULT_THRESHOLD):
    """Classify one finite numeric observation using an explicit threshold."""
    try:
        value = float(value)
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "value and threshold must be finite numbers"
        ) from exc

    if not math.isfinite(value):
        raise ValueError("value must be a finite number")

    if not math.isfinite(threshold):
        raise ValueError("threshold must be a finite number")

    score = float(model.score([[value]])[0])
    prediction = "anomaly" if score < threshold else "normal"

    return {
        "value": value,
        "prediction": prediction,
        "score": score,
        "threshold": threshold,
    }


def predict_row(model, row, threshold=DEFAULT_THRESHOLD, index=None):
    """Classify one credit-card transaction (V1–V28 + Amount)."""
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("value and threshold must be finite numbers") from exc
    if not math.isfinite(threshold):
        raise ValueError("threshold must be a finite number")

    if hasattr(row, "to_dict"):
        payload = row.to_dict()
    elif isinstance(row, dict):
        payload = row
    else:
        values = list(row)
        if len(values) != len(FEATURES_NAMES):
            raise ValueError(
                f"Expected {len(FEATURES_NAMES)} features {FEATURES_NAMES}, "
                f"got {len(values)}."
            )
        payload = dict(zip(FEATURES_NAMES, values))

    features = []
    for name in FEATURES_NAMES:
        if name not in payload:
            raise ValueError(f"Missing feature: {name}")
        try:
            value = float(payload[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        features.append(value)

    score = float(model.score([features])[0])
    return {
        "index": index,
        "prediction": "fraud" if score < threshold else "legit",
        "score": score,
        "threshold": float(threshold),
        "amount": features[-1],
    }


def predict_csv(model, path, threshold=DEFAULT_THRESHOLD):
    """Score every row in a CSV that has V1–V28 and Amount columns."""
    frame = pd.read_csv(path)
    missing = [name for name in FEATURES_NAMES if name not in frame.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(missing)
            + ". Export transactions with headers V1..V28,Amount."
        )
    return [
        predict_row(model, frame.loc[i], threshold=threshold, index=int(i))
        for i in frame.index
    ]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Score credit-card transactions with the saved Sentinel model. "
            "Pass a CSV of rows with columns V1..V28 and Amount."
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to a transaction CSV (for example tests/fixtures/creditcard_sample.csv).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fraud if score < threshold. Default is the calibrated threshold, else 0.0.",
    )

    args = parser.parse_args()
    model = load_model()
    threshold = args.threshold
    if threshold is None:
        threshold = load_calibrated_threshold(required=False)
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
    results = predict_csv(model, args.csv, threshold=threshold)

    print("Sentinel credit-card inference")
    print(f"Model: {MODEL_PATH}")
    print(f"CSV: {args.csv}")
    print(f"Rows: {len(results)}")
    print(f"Threshold: {threshold:.5f}  (fraud if score < threshold)")

    for result in results:
        print(
            f"row={result['index']} | "
            f"amount={result['amount']:.2f} | "
            f"prediction={result['prediction']} | "
            f"score={result['score']:.4f}"
        )


if __name__ == "__main__":
    main()
