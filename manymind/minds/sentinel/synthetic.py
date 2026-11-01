
"""Shared synthetic data generation for Sentinel.

Historical generators in the evaluation scripts used the same family:

- Normals: N(100, 5), 1000 observations
- Anomalies: N(anomaly_mean, 3), 20 observations
- Labels: 0 = normal, 1 = anomaly, assigned from the generating component
- Shuffle with the same Generator after both components are drawn

Differences before consolidation were only the random seed and which
anomaly means a script requested. ``train.generate_test_data`` used a
fixed seed of 99 and was not called by training.

Training data remains unlabeled normals. This module does not add
anomalies to the training set.
"""

from __future__ import annotations

import numpy as np

NORMAL_LOC = 100.0
NORMAL_SCALE = 5.0
ANOMALY_SCALE = 3.0
N_NORMAL = 1000
N_ANOMALY = 20

LABEL_NORMAL = 0
LABEL_ANOMALY = 1


def generate_normals(
    seed: int,
    n_normal: int = N_NORMAL,
    loc: float = NORMAL_LOC,
    scale: float = NORMAL_SCALE,
) -> np.ndarray:
    """Draw unlabeled normal observations for training or evaluation."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=loc, scale=scale, size=n_normal).reshape(-1, 1)


def generate_labeled_dataset(
    seed: int,
    anomaly_mean: float,
    n_normal: int = N_NORMAL,
    n_anomaly: int = N_ANOMALY,
    normal_loc: float = NORMAL_LOC,
    normal_scale: float = NORMAL_SCALE,
    anomaly_scale: float = ANOMALY_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a labeled normal/anomaly sample.

    Returns
    -------
    X :
        Shape ``(n_normal + n_anomaly, 1)``.
    y :
        ``0`` for normals and ``1`` for anomalies.
    """
    rng = np.random.default_rng(seed)

    normal = rng.normal(
        loc=normal_loc,
        scale=normal_scale,
        size=n_normal,
    )
    anomalies = rng.normal(
        loc=anomaly_mean,
        scale=anomaly_scale,
        size=n_anomaly,
    )

    X = np.concatenate([normal, anomalies])
    y = np.concatenate(
        [
            np.full(n_normal, LABEL_NORMAL, dtype=int),
            np.full(n_anomaly, LABEL_ANOMALY, dtype=int),
        ]
    )

    indices = rng.permutation(len(X))
    return X[indices].reshape(-1, 1), y[indices]
