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


def test_custom_threshold_changes_classification(model):
    value = 100.0
    score = float(model.decision_function([[value]])[0])

    # A threshold below the score should classify this value as normal.
    normal_result = predict_value(
        model,
        value,
        threshold=score - 0.01,
    )

    # A threshold above the score should classify it as anomalous.
    anomaly_result = predict_value(
        model,
        value,
        threshold=score + 0.01,
    )

    assert normal_result["prediction"] == "normal"
    assert anomaly_result["prediction"] == "anomaly"


def test_prediction_returns_threshold_used(model):
    result = predict_value(model, 100.0, threshold=0.05)

    assert result["threshold"] == 0.05


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf])
def test_non_finite_thresholds_are_rejected(model, threshold):
    with pytest.raises(ValueError, match="threshold must be a finite number"):
        predict_value(model, 100.0, threshold=threshold)


@pytest.mark.parametrize("threshold", ["invalid", None])
def test_invalid_threshold_types_are_rejected(model, threshold):
    with pytest.raises(ValueError, match="value and threshold must be finite numbers"):
        predict_value(model, 100.0, threshold=threshold)
