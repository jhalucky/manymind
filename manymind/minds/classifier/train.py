
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from manymind.minds.classifier.data import FEATURE_NAMES, load_telco
from manymind.minds.classifier.model import ChurnClassifier


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_PATH = ROOT / "artifacts" / "classifier" / "model.joblib"
METRICS_PATH = ROOT / "artifacts" / "classifier" / "metrics.json"


def generate_dataset(n_samples: int = 3000):
    """Generate a reproducible synthetic customer-churn dataset (tests / fallback)."""
    rng = np.random.default_rng(42)

    tenure = rng.integers(1, 73, n_samples)
    monthly_charges = rng.uniform(20, 150, n_samples)
    support_tickets = rng.poisson(1.5, n_samples)
    late_payments = rng.poisson(0.8, n_samples)
    usage_hours = rng.uniform(1, 100, n_samples)

    logit = (
        -1.0
        - 0.035 * (tenure - 36)
        + 0.018 * (monthly_charges - 85)
        + 0.42 * support_tickets
        + 0.65 * late_payments
        - 0.018 * (usage_hours - 50)
    )

    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    X = np.column_stack(
        [
            tenure,
            monthly_charges,
            support_tickets,
            late_payments,
            usage_hours,
        ]
    )

    missing_mask = rng.random(X.shape) < 0.01
    X[missing_mask] = np.nan

    return X, churn


def train_telco(csv_path: Path | None = None, artifact_path: Path | None = None) -> dict:
    """Train the churn model on IBM Telco and freeze holdout metrics."""
    artifact_path = Path(artifact_path) if artifact_path is not None else ARTIFACT_PATH
    X, y = load_telco(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = ChurnClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.score(X_test)

    metrics = {
        "name": "Classifier",
        "use_case": "telecom customer churn",
        "dataset": "IBM Telco Customer Churn",
        "openml_id": 42178,
        "features": list(FEATURE_NAMES),
        "n_train": int(len(X_train)),
        "n_holdout": int(len(X_test)),
        "holdout_churn_rate": float(y_test.mean()),
        "holdout": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "pr_auc": float(average_precision_score(y_test, probabilities)),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        },
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": (
            "Public IBM sample dataset, not a live carrier. "
            "Lead with ROC-AUC and churn-class F1, not accuracy."
        ),
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(artifact_path)
    metrics_path = artifact_path.parent / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Churn rate: {y.mean():.2%}")
    print(f"Accuracy: {metrics['holdout']['accuracy']:.4f}")
    print(f"Precision: {metrics['holdout']['precision']:.4f}")
    print(f"Recall: {metrics['holdout']['recall']:.4f}")
    print(f"F1-score: {metrics['holdout']['f1']:.4f}")
    print(f"ROC-AUC: {metrics['holdout']['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['holdout']['pr_auc']:.4f}")
    print("\nConfusion matrix:")
    print(np.array(metrics["holdout"]["confusion_matrix"]))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"\nSaved model to: {artifact_path}")
    print(f"Saved metrics to: {metrics_path}")
    return metrics


def main():
    train_telco()


if __name__ == "__main__":
    main()
