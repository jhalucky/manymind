
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from manymind.minds.kyc.data import load_seed
from manymind.minds.kyc.model import Guard

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_PATH = ROOT / "artifacts" / "kyc" / "model.joblib"
METRICS_PATH = ROOT / "artifacts" / "kyc" / "metrics.json"


def train_guard(seed_path: Path | None = None, artifact_path: Path | None = None) -> dict:
    artifact_path = Path(artifact_path) if artifact_path is not None else ARTIFACT_PATH
    X, y = load_seed(seed_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Guard()
    model.fit(X_train, y_train)
    probabilities = model.score(X_test)
    predictions = (probabilities >= model.threshold).astype(int)

    metrics = {
        "name": "Guard",
        "use_case": "KYC / bait SMS check",
        "dataset": "hand-written seed (not a bank dump)",
        "n_train": int(len(X_train)),
        "n_holdout": int(len(X_test)),
        "holdout_bait_rate": float(y_test.mean()),
        "threshold": model.threshold,
        "holdout": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        },
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": (
            "Trained on synthetic seed templates. "
            "A bait label is a warning, not proof of a crime. "
            "Never share OTP, PIN, or Aadhaar."
        ),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(artifact_path)
    metrics_path = artifact_path.parent / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    hold = metrics["holdout"]
    print(f"Training messages: {len(X_train)}")
    print(f"Holdout messages: {len(X_test)}")
    print(
        f"Holdout  P={hold['precision']:.3f}  R={hold['recall']:.3f}  "
        f"F1={hold['f1']:.3f}  ROC-AUC={hold['roc_auc']:.3f}"
    )
    print(f"Model saved to: {artifact_path}")
    print(f"Metrics saved to: {metrics_path}")
    return metrics


if __name__ == "__main__":
    train_guard()
