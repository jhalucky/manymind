
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


PROJECT_ROOT = Path("/home/lucky/ManyMind")
DATA_DIR = PROJECT_ROOT / "data" / "sentinel"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "sentinel"


def generate_training_data():
    """Generate normal observations for model training."""
    rng = np.random.default_rng(42)

    normal = rng.normal(
        loc=100,
        scale=5,
        size=1000,
    )

    return normal.reshape(-1, 1)



def generate_test_data(anomaly_mean):
    """Generate a test dataset with adjustable anomaly difficulty."""
    rng = np.random.default_rng(99)

    normal = rng.normal(
        loc=100,
        scale=5,
        size=1000,
    )

    anomalies = rng.normal(
        loc=anomaly_mean,
        scale=3,
        size=20,
    )

    X_test = np.concatenate([normal, anomalies])

    y_test = np.concatenate([
        np.zeros(len(normal), dtype=int),
        np.ones(len(anomalies), dtype=int),
    ])

    indices = rng.permutation(len(X_test))

    return X_test[indices].reshape(-1, 1), y_test[indices]





def train():
    """Train Sentinel on normal data and save the selected artifact."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    X_train = generate_training_data()

    pd.DataFrame({
        "value": X_train.flatten(),
    }).to_csv(DATA_DIR / "train.csv", index=False)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42,
    )

    model.fit(X_train)

    model_path = ARTIFACT_DIR / "model.joblib"
    joblib.dump(model, model_path)

    print(f"Training observations: {len(X_train)}")
    print(f"Contamination: {model.contamination}")
    print(f"Trees: {len(model.estimators_)}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train()
