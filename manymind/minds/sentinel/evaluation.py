
"""Shared Sentinel evaluation protocol: splits, scoring, and metrics.

An observation is labeled anomalous when::

    score < threshold

``score`` is Sentinel's Isolation Forest ``decision_function`` value
(higher is more normal). The default threshold is ``0.0``. Native
``predict()`` labels (``1`` inlier / ``-1`` outlier) are a consistency
check for that default only; they do not apply a custom threshold.

These helpers are for synthetic experimentation. They do not establish
real-world performance.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix

from manymind.minds.sentinel.model import DEFAULT_THRESHOLD
from manymind.minds.sentinel.synthetic import generate_labeled_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SENTINEL_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "sentinel"
SENTINEL_DATA_DIR = PROJECT_ROOT / "data" / "sentinel"
SENTINEL_RUNS_DIR = SENTINEL_DATA_DIR / "runs"
THRESHOLD_CONFIG_PATH = SENTINEL_ARTIFACT_DIR / "threshold.json"


class ThresholdConfigError(ValueError):
    """Raised when a calibrated threshold configuration is missing or invalid."""

TRAIN_SEED = 42
CALIBRATION_SEEDS = (3101, 3102, 3103, 3104, 3105)
HOLDOUT_SEEDS = (9101, 9102, 9103, 9104, 9105)
ROBUSTNESS_SEEDS = (2026, 2027, 2028, 2029, 2030)

CALIBRATION_ANOMALY_MEANS = (120, 115, 110)
HOLDOUT_ANOMALY_MEANS = (120, 115, 110)
ROBUSTNESS_ANOMALY_MEANS = (150, 120, 115, 110)

TARGET_RECALL = 0.90
COMPARISON_THRESHOLDS = (0.0, 0.02, 0.05, 0.08, 0.10)


def assigned_seed_groups() -> dict[str, set[int]]:
    return {
        "train": {TRAIN_SEED},
        "calibration": set(CALIBRATION_SEEDS),
        "holdout": set(HOLDOUT_SEEDS),
        "robustness": set(ROBUSTNESS_SEEDS),
    }


def validate_disjoint_splits(
    groups: dict[str, Iterable[int]] | None = None,
) -> None:
    """Reject overlapping train / calibration / holdout / robustness seeds."""
    resolved = {
        name: set(seeds)
        for name, seeds in (groups or assigned_seed_groups()).items()
    }
    names = list(resolved)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = resolved[left] & resolved[right]
            if overlap:
                raise ValueError(
                    "Sentinel evaluation seeds overlap between "
                    f"{left} and {right}: {sorted(overlap)}"
                )


def labels_from_scores(
    scores: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Return 1 for anomaly and 0 for normal using ``score < threshold``."""
    return (np.asarray(scores) < threshold).astype(int)


def native_predict_matches_default_threshold(model: Any, X: Any) -> bool:
    """Return True if ``predict()`` matches ``score < 0.0`` on ``X``."""
    scores = model.score(X)
    expected = np.where(scores < DEFAULT_THRESHOLD, -1, 1)
    return np.array_equal(model.predict(X), expected)


def assert_native_predict_matches_default_threshold(model: Any, X: Any) -> None:
    if not native_predict_matches_default_threshold(model, X):
        raise RuntimeError(
            "IsolationForest predict() does not match score < "
            f"{DEFAULT_THRESHOLD} on the provided observations."
        )


def calculate_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = labels_from_scores(scores, threshold)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    fpr = fp / (fp + tn) if fp + tn else 0.0

    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
    }


def iter_labeled_splits(
    seeds: Sequence[int],
    anomaly_means: Sequence[float],
    **generator_kwargs: Any,
):
    for anomaly_mean in anomaly_means:
        for seed in seeds:
            X, y = generate_labeled_dataset(
                seed=seed,
                anomaly_mean=anomaly_mean,
                **generator_kwargs,
            )
            yield float(anomaly_mean), int(seed), X, y


def score_splits(
    model: Any,
    seeds: Sequence[int],
    anomaly_means: Sequence[float],
    **generator_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_scores = []
    all_labels = []
    all_means = []

    for anomaly_mean, _seed, X, y in iter_labeled_splits(
        seeds,
        anomaly_means,
        **generator_kwargs,
    ):
        all_scores.append(model.score(X))
        all_labels.append(y)
        all_means.append(np.full(len(y), anomaly_mean, dtype=float))

    return (
        np.concatenate(all_scores),
        np.concatenate(all_labels),
        np.concatenate(all_means),
    )


def coerce_finite_threshold(value: Any, *, source: str) -> float:
    """Convert a configuration value to a finite threshold without rounding."""
    if isinstance(value, bool) or value is None:
        raise ThresholdConfigError(
            f"Invalid threshold in {source}: {value!r} is not a finite number."
        )

    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {
            "",
            "nan",
            "+nan",
            "-nan",
            "inf",
            "+inf",
            "-inf",
            "infinity",
            "+infinity",
            "-infinity",
        }:
            raise ThresholdConfigError(
                f"Invalid threshold in {source}: {value!r} is not a finite number."
            )

    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ThresholdConfigError(
            f"Invalid threshold in {source}: {value!r} cannot be converted "
            "to a finite number."
        ) from exc

    if not math.isfinite(threshold):
        raise ThresholdConfigError(
            f"Invalid threshold in {source}: {value!r} is not a finite number."
        )

    return threshold


def load_calibrated_threshold(
    path: str | Path | None = None,
    *,
    required: bool = True,
) -> float | None:
    """Read a previously saved threshold without modifying or rounding it.

    If ``required`` is true, a missing file is an error. If ``required`` is
    false, a missing file returns ``None``. A file that exists but is
    invalid always fails, including optional loads.
    """
    config_path = Path(path) if path is not None else THRESHOLD_CONFIG_PATH

    if not config_path.is_file():
        if required:
            raise ThresholdConfigError(
                f"Required threshold configuration not found: {config_path}"
            )
        return None

    try:
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as exc:
        raise ThresholdConfigError(
            f"Threshold configuration is not valid JSON: {config_path}"
        ) from exc

    if not isinstance(config, dict):
        raise ThresholdConfigError(
            f"Threshold configuration must be a JSON object: {config_path}"
        )

    if "threshold" not in config:
        raise ThresholdConfigError(
            f"Threshold configuration is missing the 'threshold' key: {config_path}"
        )

    return coerce_finite_threshold(config["threshold"], source=str(config_path))


def holdout_thresholds(
    extra: Sequence[float] = (0.05,),
    config_path: str | Path | None = None,
    *,
    require_calibrated: bool = True,
) -> list[float]:
    """Default 0.0, extra comparison values, and the saved candidate.

    Holdout evaluation requires the saved candidate. Calibration-set
    comparison may omit it when no configuration file exists.
    """
    values = [float(DEFAULT_THRESHOLD), *[float(item) for item in extra]]
    calibrated = load_calibrated_threshold(
        config_path,
        required=require_calibrated,
    )
    if calibrated is not None:
        values.append(calibrated)

    unique: list[float] = []
    for value in values:
        if not any(np.isclose(value, existing) for existing in unique):
            unique.append(value)
    return unique


def new_run_directory(script_name: str) -> Path:
    """Create a unique output directory under ``data/sentinel/runs``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    directory = SENTINEL_RUNS_DIR / f"{script_name}_{stamp}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_writable_path(path: str | Path) -> Path:
    """Return ``path`` after creating parents, refusing to overwrite."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing file: {destination}\n"
            "Pass a new --output/--output-dir path if you want to keep this file."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def default_run_output(script_name: str, filename: str) -> Path:
    return resolve_writable_path(new_run_directory(script_name) / filename)


validate_disjoint_splits()
