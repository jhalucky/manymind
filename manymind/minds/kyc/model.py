
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from manymind.core.base import BaseMind

DEFAULT_THRESHOLD = 0.5

ADVICE = (
    "Do not tap the link or send money. Open the bank or UIDAI app yourself. "
    "Never paste OTP, UPI PIN, or Aadhaar here."
)

_URL = re.compile(r"https?://|\bw\.w|\bbit\.ly\b|\btinyurl\b", re.I)
_KYC = re.compile(r"\bkyc\b|\baadhaar\b|\buidai\b|\bre-?kyc\b", re.I)
_URGENCY = re.compile(
    r"\b(urgent|immediately|within \d+ hours?|last chance|today|blocked|deactivat)",
    re.I,
)
_ASK_SECRET = re.compile(
    r"\b(share otp|send otp|enter otp|give otp|share pin|enter pin|cvv|do not tell)\b",
    re.I,
)
_PAY = re.compile(r"\b(pay|fee|deposit|transfer|send rs|send ₹)\b", re.I)


def rule_flags(text: str) -> list[str]:
    flags: list[str] = []
    if _URL.search(text):
        flags.append("contains_link")
    if _KYC.search(text):
        flags.append("kyc_or_aadhaar")
    if _URGENCY.search(text):
        flags.append("urgency")
    if _ASK_SECRET.search(text):
        flags.append("asks_secret")
    if _PAY.search(text) and _KYC.search(text):
        flags.append("pay_for_kyc")
    return flags


def _as_texts(X: Any) -> list[str]:
    if isinstance(X, str):
        return [X]
    return [str(item).strip() for item in np.asarray(X, dtype=object).ravel()]


class Guard(BaseMind):
    """Flag KYC / bait SMS from pasted text.

    Trained on a hand-written seed, not on a bank dump. A 'bait' label
    is a warning to verify in the official app, not a legal finding.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = float(threshold)
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=8000,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        self.is_fitted = False

    def fit(self, X: Any, y: Any = None) -> "Guard":
        if y is None:
            raise ValueError("Training labels y are required.")
        self.pipeline.fit(_as_texts(X), y)
        self.is_fitted = True
        return self

    def score(self, X: Any) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")
        texts = _as_texts(X)
        proba = self.pipeline.predict_proba(texts)
        classes = list(self.pipeline.named_steps["clf"].classes_)
        bait_index = classes.index(1)
        model_p = proba[:, bait_index]
        blended = []
        for text, p in zip(texts, model_p):
            flags = rule_flags(text)
            bump = 0.0
            if "asks_secret" in flags or "pay_for_kyc" in flags:
                bump = 0.35
            elif "contains_link" in flags and "kyc_or_aadhaar" in flags:
                bump = 0.25
            elif "contains_link" in flags and "urgency" in flags:
                bump = 0.2
            blended.append(float(min(1.0, p + bump)))
        return np.asarray(blended, dtype=float)

    def predict(self, X: Any) -> np.ndarray:
        return (self.score(X) >= self.threshold).astype(int)

    def explain(self, text: str) -> dict:
        probability = float(self.score([text])[0])
        flags = rule_flags(text)
        bait = probability >= self.threshold
        return {
            "text": text,
            "prediction": "bait" if bait else "ok",
            "bait_probability": probability,
            "threshold": self.threshold,
            "flags": flags,
            "advice": ADVICE if bait else "Looks closer to a normal alert. Still never share OTP or PIN.",
        }

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "threshold": self.threshold}, destination)

    def load(self, path: str | Path) -> "Guard":
        destination = Path(path)
        if not destination.is_file():
            raise FileNotFoundError(f"Model artifact not found: {destination}")
        payload = joblib.load(destination)
        if isinstance(payload, Pipeline):
            self.pipeline = payload
            self.threshold = DEFAULT_THRESHOLD
        elif isinstance(payload, dict) and "pipeline" in payload:
            self.pipeline = payload["pipeline"]
            self.threshold = float(payload.get("threshold", DEFAULT_THRESHOLD))
        else:
            raise ValueError("Artifact does not contain a Guard pipeline.")
        if not isinstance(self.pipeline, Pipeline):
            raise ValueError("Artifact does not contain a valid sklearn Pipeline.")
        self.is_fitted = True
        return self
