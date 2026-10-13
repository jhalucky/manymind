import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from manymind.minds.sentinel.predict import predict_value


@pytest.fixture
def model():
    rng = np.random.default_rng(42)

    normal_data = rng.normal(
        loc=100,
        scale=5,
        size=1000,
    ).reshape(-1, 1)

    fitted_model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42,
    )

    fitted_model.fit(normal_data)
    return fitted_model


def test_center_value_is_normal(model):
    result = predict_value(model, 100.0)

    assert result["prediction"] == "normal"
    assert result["value"] == 100.0
    assert np.isfinite(result["score"])


def test_extreme_value_is_anomaly(model):
    result = predict_value(model, 150.0)

    assert result["prediction"] == "anomaly"
    assert np.isfinite(result["score"])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_rejected(model, value):
    with pytest.raises(ValueError):
        predict_value(model, value)
