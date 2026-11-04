from pathlib import Path

import pandas as pd

SEED_PATH = Path(__file__).with_name("seed.csv")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPAM_PATH = PROJECT_ROOT / "data" / "guard" / "spam.csv"


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


def clean_spam_csv(path: str | Path | None = None) -> Path:
    """Keep only text and label. Merge overflow columns from the UCI export."""
    csv_path = Path(path) if path is not None else SPAM_PATH
    if not csv_path.is_file():
        raise FileNotFoundError(f"Spam dataset not found: {csv_path}")

    try:
        frame = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        frame = pd.read_csv(csv_path, encoding="latin-1")

    if {"text", "label"}.issubset(frame.columns) and frame.shape[1] == 2:
        cleaned = frame[["text", "label"]].copy()
    elif {"v1", "v2"}.issubset(frame.columns):
        text = frame["v2"].fillna("").astype(str)
        for column in frame.columns:
            if column in {"v1", "v2"}:
                continue
            extra = frame[column].fillna("").astype(str)
            text = text.str.cat(extra, sep=" ")
        label = frame["v1"].astype(str).str.strip().str.lower().map({"ham": 0, "spam": 1})
        cleaned = pd.DataFrame({"text": text.str.replace(r"\s+", " ", regex=True).str.strip(), "label": label})
    else:
        raise ValueError(
            f"Expected columns v1,v2 or text,label in {csv_path}, got {list(frame.columns)}"
        )

    cleaned = cleaned.dropna(subset=["text", "label"])
    cleaned = cleaned[cleaned["text"].str.len() > 0]
    cleaned["label"] = cleaned["label"].astype(int)
    if set(cleaned["label"].unique()) - {0, 1}:
        raise ValueError("Spam labels must map to 0 (ham) or 1 (spam).")
    cleaned.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def load_spam(path: str | Path | None = None) -> tuple[pd.Series, pd.Series]:
    csv_path = clean_spam_csv(path)
    frame = pd.read_csv(csv_path, encoding="utf-8")
    return frame["text"].astype(str), frame["label"].astype(int)


def load_training_messages() -> tuple[pd.Series, pd.Series]:
    """SMS spam corpus plus the KYC seed so Guard still sees bait templates."""
    spam_text, spam_y = load_spam()
    seed_text, seed_y = load_seed()
    text = pd.concat([spam_text, seed_text], ignore_index=True)
    label = pd.concat([spam_y, seed_y], ignore_index=True)
    return text, label
