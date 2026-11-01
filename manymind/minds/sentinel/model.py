
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import IsolationForest

from manymind.core.base import BaseMind

# IsolationForest decision_function boundary used by the public CLI.
DEFAULT_THRESHOLD = 0.0


class Sentinel(BaseMind):
    """Detect unusual numerical observations with Isolation Forest.

    This model is intended for experimentation with synthetic data,
    not for making real-world operational decisions.

    Scoring convention:
    - ``score`` returns IsolationForest ``decision_function`` values.
      Higher scores are more normal; values below 0.0 are anomalous
      at the default threshold.
    - ``predict`` returns IsolationForest labels: ``1`` (normal) and
      ``-1`` (anomaly), equivalent to ``score < 0.0``.
    - Custom thresholds are a CLI concern and are not stored on the model.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.01,
        random_state: int = 42,
    ) -> None:
        self.estimator = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.is_fitted = False

    @property
    def contamination(self) -> float:
        return float(self.estimator.contamination)

    def fit(self, X: Any, y: Any = None) -> "Sentinel":
        self.estimator.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: Any) -> Any:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        return self.estimator.predict(X)

    def score(self, X: Any) -> Any:
        """Return IsolationForest decision_function scores.

        These are not class probabilities. They are ``score_samples``
        shifted by the fitted offset so that the native decision
        boundary is ``0.0``.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")

        return self.estimator.decision_function(X)

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, destination)

    def load(self, path: str | Path) -> "Sentinel":
        destination = Path(path)

        if not destination.is_file():
            raise FileNotFoundError(f"Model artifact not found: {destination}")

        estimator = joblib.load(destination)

        if not isinstance(estimator, IsolationForest):
            raise ValueError("Artifact does not contain a valid IsolationForest.")

        self.estimator = estimator
        self.is_fitted = hasattr(estimator, "offset_")

        if not self.is_fitted:
            raise ValueError("The loaded IsolationForest has not been fitted.")

        return self
