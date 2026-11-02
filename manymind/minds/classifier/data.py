from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = PROJECT_ROOT / "data" / "classifier" / "telco" / "telco.csv"

FEATURE_NAMES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def load_telco(path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load IBM Telco Customer Churn. Label 1 = churned."""
    csv_path = Path(path) if path is not None else CSV_PATH
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Telco dataset not found: {csv_path}\n"
            "Download it with the commands in README.md (Datasets)."
        )

    df = pd.read_csv(csv_path)
    missing = [name for name in FEATURE_NAMES + ["Churn"] if name not in df.columns]
    if missing:
        raise ValueError(f"Telco CSV is missing columns: {missing}")

    frame = df[FEATURE_NAMES].copy()
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    y = df["Churn"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if y.isna().any():
        raise ValueError("Churn column must contain Yes/No values.")
    return frame, y.astype(int)
