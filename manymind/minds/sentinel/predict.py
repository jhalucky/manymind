
from pathlib import Path
import argparse
import math

import joblib


PROJECT_ROOT = Path("/home/lucky/ManyMind")
MODEL_PATH = PROJECT_ROOT / "artifacts" / "sentinel" / "model.joblib"


def load_model():
    """Load the persisted Sentinel model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Sentinel model not found: {MODEL_PATH}\n"
            "Train a model and save the artifact first."
        )

    return joblib.load(MODEL_PATH)


def predict_value(model, value):
    """Classify one numeric observation using a loaded model."""
    if not math.isfinite(value):
        raise ValueError("value must be a finite number")

    prediction = int(model.predict([[value]])[0])
    score = float(model.decision_function([[value]])[0])

    return {
        "value": value,
        "prediction": "anomaly" if prediction == -1 else "normal",
        "score": score,
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

    args = parser.parse_args()
    model = load_model()

    print("Sentinel inference")
    print(f"Model: {MODEL_PATH}")
    print(f"Contamination: {model.contamination}")

    for value in args.values:
        result = predict_value(model, value)

        print(
            f"value={result['value']:.2f} | "
            f"prediction={result['prediction']} | "
            f"score={result['score']:.4f}"
        )


if __name__ == "__main__":
    main()

