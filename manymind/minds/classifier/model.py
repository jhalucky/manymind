
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from manymind.core.base import BaseMind


def _as_frame(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(np.asarray(X))


class ChurnClassifier(BaseMind):
    """Predict customer churn using supervised learning.

    Production training uses the public IBM Telco dataset. The same
    wrapper still accepts numeric arrays so unit tests stay lightweight.
    """

    def __init__(self) -> None:
        numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        self.pipeline = Pipeline(
            [
                (
                    "preprocess",
                    ColumnTransformer(
                        [
                            (
                                "num",
                                numeric,
                                make_column_selector(dtype_include=np.number),
                            ),
                            (
                                "cat",
                                categorical,
                                make_column_selector(dtype_exclude=np.number),
                            ),
                        ]
                    ),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        self.is_fitted = False

    def fit(self, X: Any, y: Any = None) -> "ChurnClassifier":
        if y is None:
            raise ValueError("Training labels y are required.")

        self.pipeline.fit(_as_frame(X), y)
        self.is_fitted = True
        return self

    def predict(self, X: Any) -> Any:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        return self.pipeline.predict(_as_frame(X))

    def score(self, X: Any) -> Any:
        """Return the predicted probability of churn (class 1)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")

        classifier = self.pipeline.named_steps["classifier"]
        classes = list(classifier.classes_)

        if 1 not in classes:
            raise RuntimeError("The fitted model has no churn class (1).")

        churn_index = classes.index(1)
        return self.pipeline.predict_proba(_as_frame(X))[:, churn_index]

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, destination)

    def load(self, path: str | Path) -> "ChurnClassifier":
        destination = Path(path)

        if not destination.is_file():
            raise FileNotFoundError(f"Model artifact not found: {destination}")

        self.pipeline = joblib.load(destination)

        if not isinstance(self.pipeline, Pipeline):
            raise ValueError("Artifact does not contain a valid sklearn Pipeline.")

        if "classifier" not in self.pipeline.named_steps:
            raise ValueError("Pipeline is missing its classifier.")

        classifier = self.pipeline.named_steps["classifier"]

        if not isinstance(classifier, RandomForestClassifier):
            raise ValueError("Pipeline does not contain a RandomForestClassifier.")

        self.is_fitted = hasattr(classifier, "classes_")

        if not self.is_fitted:
            raise ValueError("The loaded classifier has not been fitted.")

        return self
