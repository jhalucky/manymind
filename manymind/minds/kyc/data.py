from pathlib import Path

import pandas as pd

SEED_PATH = Path(__file__).with_name("seed.csv")


def load_seed(path: str | Path | None = None) -> tuple[pd.Series, pd.Series]:
    """Load the KYC seed. label 1 = bait, 0 = okay."""
    csv_path = Path(path) if path is not None else SEED_PATH
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"KYC seed not found: {csv_path}\n"
            "Run: python -m manymind.minds.kyc.build_seed"
        )
    frame = pd.read_csv(csv_path)
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Seed CSV missing columns: {sorted(missing)}")
    text = frame["text"].astype(str).str.strip()
    label = frame["label"].astype(int)
    if set(label.unique()) - {0, 1}:
        raise ValueError("Seed labels must be 0 (ok) or 1 (bait).")
    if text.eq("").any():
        raise ValueError("Seed has empty messages.")
    return text, label
