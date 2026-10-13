
import argparse
import math
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "sentinel" / "model.joblib"

# IsolationForest's existing decision boundary on decision_function scores.
DEFAULT_THRESHOLD = 0.0


def load_model():
    """Load the persisted Sentinel model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Sentinel model not found: {MODEL_PATH}\n"
            "Train a model and save the artifact first."
        )

    return joblib.load(MODEL_PATH)


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

    score = float(model.decision_function([[value]])[0])
    prediction = "anomaly" if score < threshold else "normal"

    return {
        "value": value,
        "prediction": prediction,
        "score": score,
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with the saved Sentinel model."
    )

    parser.add_argument(
        "values",
        nargs="+",
        type=float,
        help="One or more numeric observations to classify.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Decision threshold for anomaly scores. "
            "Defaults to 0.0, the model's existing decision boundary."
        ),
    )

    args = parser.parse_args()
    model = load_model()

    print("Sentinel inference")
    print(f"Model: {MODEL_PATH}")
    print(f"Contamination: {model.contamination}")
    print(f"Threshold: {args.threshold:.5f}")

    for value in args.values:
        result = predict_value(
            model,
            value,
            threshold=args.threshold,
        )

        print(
            f"value={result['value']:.2f} | "
            f"prediction={result['prediction']} | "
            f"score={result['score']:.4f}"
        )


if __name__ == "__main__":
    main()
