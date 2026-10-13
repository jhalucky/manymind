
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from manymind.minds.classifier.model import ChurnClassifier


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_PATH = ROOT / "artifacts" / "classifier" / "model.joblib"

FEATURE_NAMES = [
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "late_payments",
    "usage_hours",
]


def generate_dataset(n_samples: int = 3000):
    """Generate a reproducible synthetic customer-churn dataset."""
    rng = np.random.default_rng(42)

    tenure = rng.integers(1, 73, n_samples)
    monthly_charges = rng.uniform(20, 150, n_samples)
    support_tickets = rng.poisson(1.5, n_samples)
    late_payments = rng.poisson(0.8, n_samples)
    usage_hours = rng.uniform(1, 100, n_samples)

    # Synthetic rule: higher support burden and late payments,
    # combined with shorter tenure and lower usage, increase churn risk.
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

    # Simulate a small amount of missing numeric data.
    missing_mask = rng.random(X.shape) < 0.01
    X[missing_mask] = np.nan

    return X, churn


def main():
    X, y = generate_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = ChurnClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Churn rate: {y.mean():.2%}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score: {f1_score(y_test, predictions, zero_division=0):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(ARTIFACT_PATH)
    print(f"\nSaved model to: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
