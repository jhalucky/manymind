
import argparse
from pathlib import Path

import pandas as pd

from manymind.minds.classifier.data import FEATURE_NAMES
from manymind.minds.classifier.model import ChurnClassifier
from manymind.minds.classifier.train import ARTIFACT_PATH


def load_model(path: str | Path | None = None) -> ChurnClassifier:
    model_path = Path(path) if path is not None else ARTIFACT_PATH
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Classifier model not found: {model_path}\n"
            "Train it with: python -m manymind.minds.classifier.train"
        )
    return ChurnClassifier().load(model_path)


def predict_frame(model: ChurnClassifier, frame: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(missing)
            + ". Export customers with the IBM Telco headers."
        )
    X = frame[FEATURE_NAMES].copy()
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
    labels = model.predict(X)
    probabilities = model.score(X)
    out = frame.copy()
    out["churn_probability"] = probabilities
    out["prediction"] = ["churn" if int(label) == 1 else "retain" for label in labels]
    return out


def predict_csv(model: ChurnClassifier, path: str | Path) -> pd.DataFrame:
    return predict_frame(model, pd.read_csv(path))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Score telco customers with the saved churn model. "
            "Pass a CSV with IBM Telco feature columns."
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to a customer CSV (for example tests/fixtures/telco_sample.csv).",
    )
    args = parser.parse_args()

    model = load_model()
    results = predict_csv(model, args.csv)

    print("Classifier telco inference")
    print(f"Model: {ARTIFACT_PATH}")
    print(f"CSV: {args.csv}")
    print(f"Rows: {len(results)}")

    for index, row in results.iterrows():
        print(
            f"row={index} | "
            f"tenure={row['tenure']} | "
            f"prediction={row['prediction']} | "
            f"p_churn={row['churn_probability']:.3f}"
        )


if __name__ == "__main__":
    main()
