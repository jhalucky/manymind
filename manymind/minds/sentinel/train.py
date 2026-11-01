
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from manymind.minds.sentinel.data import load_creditcard
from manymind.minds.sentinel.evaluation import TRAIN_SEED, calculate_metrics
from manymind.minds.sentinel.model import Sentinel
from manymind.minds.sentinel.synthetic import generate_labeled_dataset, generate_normals


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "sentinel"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "sentinel"


def generate_training_data():
    """Generate unlabeled normal observations for model training."""
    return generate_normals(seed=TRAIN_SEED)


def generate_test_data(anomaly_mean, seed=99):
    """Generate a labeled dataset with adjustable anomaly difficulty."""
    return generate_labeled_dataset(seed=seed, anomaly_mean=anomaly_mean)


def train(data_dir: Path | None = None, artifact_dir: Path | None = None):
    """Train Sentinel on synthetic normal data (used by unit tests)."""
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    artifact_dir = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR

    data_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    X_train = generate_training_data()

    pd.DataFrame({"value": X_train.flatten()}).to_csv(data_dir / "train.csv", index=False)

    model = Sentinel()
    model.fit(X_train)

    model_path = artifact_dir / "model.joblib"
    model.save(model_path)

    print(f"Training observations: {len(X_train)}")
    print(f"Contamination: {model.contamination}")
    print(f"Trees: {len(model.estimator.estimators_)}")
    print(f"Model saved to: {model_path}")


def _choose_threshold(scores: np.ndarray, y: np.ndarray) -> dict:
    """Pick the threshold with maximum F1 on calibration scores."""
    quantiles = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 201)))
    best: dict | None = None
    for threshold in quantiles:
        metrics = calculate_metrics(y, scores, float(threshold))
        if best is None or metrics["f1"] > best["f1"]:
            best = metrics
        elif best is not None and metrics["f1"] == best["f1"] and metrics["precision"] > best["precision"]:
            best = metrics
    if best is None:
        raise RuntimeError("Could not select a calibration threshold.")
    return best


def train_creditcard(
    csv_path: Path | None = None,
    artifact_dir: Path | None = None,
) -> dict:
    """Train Sentinel on the ULB credit-card dataset and freeze holdout metrics."""
    artifact_dir = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_creditcard(csv_path) if csv_path is not None else load_creditcard()
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_cal, X_hold, y_cal, y_hold = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=42, stratify=y_rest
    )

    contamination = float(y_train.mean())
    model = Sentinel(contamination=contamination, random_state=42)
    model.fit(X_train)

    cal_scores = np.asarray(model.score(X_cal), dtype=float)
    y_cal_np = np.asarray(y_cal, dtype=int)
    selected = _choose_threshold(cal_scores, y_cal_np)
    threshold = float(selected["threshold"])

    hold_scores = np.asarray(model.score(X_hold), dtype=float)
    y_hold_np = np.asarray(y_hold, dtype=int)
    holdout = calculate_metrics(y_hold_np, hold_scores, threshold)
    holdout["roc_auc"] = float(roc_auc_score(y_hold_np, -hold_scores))
    holdout["pr_auc"] = float(average_precision_score(y_hold_np, -hold_scores))

    model_path = artifact_dir / "model.joblib"
    threshold_path = artifact_dir / "threshold.json"
    metrics_path = artifact_dir / "metrics.json"

    model.save(model_path)

    threshold_payload = {
        "threshold": threshold,
        "selection_metric": "maximum_f1",
        "dataset": "ULB creditcard (OpenML 1597 / public benchmark)",
        "n_train": int(len(X_train)),
        "n_calibration": int(len(X_cal)),
        "n_holdout": int(len(X_hold)),
        "train_fraud_rate": contamination,
        "calibration_metrics": selected,
        "sklearn_version": sklearn.__version__,
    }
    threshold_path.write_text(json.dumps(threshold_payload, indent=2) + "\n", encoding="utf-8")

    metrics_payload = {
        "name": "Sentinel",
        "use_case": "credit-card fraud review",
        "dataset": "ULB creditcard",
        "openml_id": 1597,
        "features": ["V1-V28", "Amount"],
        "n_train": int(len(X_train)),
        "n_calibration": int(len(X_cal)),
        "n_holdout": int(len(X_hold)),
        "holdout_fraud_count": int(y_hold_np.sum()),
        "threshold": threshold,
        "threshold_rule": "score < threshold -> fraud",
        "selection_metric": "maximum_f1 on calibration",
        "holdout": holdout,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": (
            "Public benchmark dataset. Not a live bank deployment. "
            "PR-AUC and precision/recall at the frozen threshold are the "
            "headline numbers, not accuracy."
        ),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")

    print(f"Training observations: {len(X_train)}")
    print(f"Calibration observations: {len(X_cal)}")
    print(f"Holdout observations: {len(X_hold)}")
    print(f"Contamination: {contamination:.6f}")
    print(f"Calibrated threshold: {threshold:.5f}")
    print(
        "Holdout  "
        f"ROC-AUC={holdout['roc_auc']:.4f}  "
        f"PR-AUC={holdout['pr_auc']:.4f}  "
        f"P={holdout['precision']:.3f}  "
        f"R={holdout['recall']:.3f}  "
        f"F1={holdout['f1']:.3f}"
    )
    print(f"Model saved to: {model_path}")
    print(f"Threshold saved to: {threshold_path}")
    print(f"Metrics saved to: {metrics_path}")
    return metrics_payload


if __name__ == "__main__":
    train_creditcard()
