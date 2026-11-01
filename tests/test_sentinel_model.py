
import joblib
import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from manymind.core.base import BaseMind
from manymind.minds.sentinel.model import DEFAULT_THRESHOLD, Sentinel
from manymind.minds.sentinel import predict as predict_module
from manymind.minds.sentinel.predict import load_model, predict_value


@pytest.fixture
def training_data():
    rng = np.random.default_rng(42)
    return rng.normal(loc=100, scale=5, size=1000).reshape(-1, 1)


def test_sentinel_is_a_base_mind():
    assert issubclass(Sentinel, BaseMind)
    assert isinstance(Sentinel(), BaseMind)


def test_fit_returns_model(training_data):
    model = Sentinel()
    result = model.fit(training_data)

    assert result is model
    assert model.is_fitted


def test_predict_returns_isolation_forest_labels(training_data):
    model = Sentinel().fit(training_data)
    predictions = model.predict(training_data[:10])

    assert len(predictions) == 10
    assert set(predictions).issubset({-1, 1})


def test_score_returns_decision_function_values(training_data):
    model = Sentinel().fit(training_data)
    scores = model.score(np.array([[100.0], [150.0]]))

    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
    assert scores[0] > scores[1]


def test_default_predict_matches_zero_threshold(training_data):
    model = Sentinel().fit(training_data)
    X = np.array([[100.0], [115.0], [150.0]])

    scores = model.score(X)
    predictions = model.predict(X)
    expected = np.where(scores < DEFAULT_THRESHOLD, -1, 1)

    assert DEFAULT_THRESHOLD == 0.0
    np.testing.assert_array_equal(predictions, expected)


def test_predict_requires_fitted_model(training_data):
    with pytest.raises(RuntimeError, match="fitted"):
        Sentinel().predict(training_data[:5])


def test_score_requires_fitted_model(training_data):
    with pytest.raises(RuntimeError, match="fitted"):
        Sentinel().score(training_data[:5])


def test_cannot_save_unfitted_model(tmp_path):
    with pytest.raises(RuntimeError, match="unfitted"):
        Sentinel().save(tmp_path / "model.joblib")


def test_save_and_load_preserve_predictions(tmp_path, training_data):
    model = Sentinel().fit(training_data)
    artifact = tmp_path / "sentinel" / "model.joblib"
    X = np.array([[100.0], [115.0], [150.0]])

    model.save(artifact)
    restored = Sentinel().load(artifact)

    np.testing.assert_array_equal(model.predict(X), restored.predict(X))
    np.testing.assert_allclose(model.score(X), restored.score(X))


def test_load_wraps_existing_isolation_forest_artifact(tmp_path, training_data):
    estimator = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42,
    )
    estimator.fit(training_data)

    artifact = tmp_path / "model.joblib"
    joblib.dump(estimator, artifact)

    loaded = Sentinel().load(artifact)
    X = np.array([[100.0], [150.0]])

    np.testing.assert_array_equal(loaded.predict(X), estimator.predict(X))
    np.testing.assert_allclose(loaded.score(X), estimator.decision_function(X))


def test_load_model_returns_fitted_sentinel(tmp_path, monkeypatch, training_data):
    artifact = tmp_path / "model.joblib"
    Sentinel().fit(training_data).save(artifact)
    monkeypatch.setattr(predict_module, "MODEL_PATH", artifact)

    model = load_model()

    assert isinstance(model, Sentinel)
    assert model.is_fitted
    assert predict_value(model, 100.0)["prediction"] == "normal"


def test_cli_accepts_values_and_custom_threshold(
    tmp_path,
    monkeypatch,
    capsys,
    training_data,
):
    artifact = tmp_path / "model.joblib"
    Sentinel().fit(training_data).save(artifact)
    monkeypatch.setattr(predict_module, "MODEL_PATH", artifact)
    monkeypatch.setattr(
        "sys.argv",
        ["predict", "100", "150", "--threshold", "0.05"],
    )

    predict_module.main()
    output = capsys.readouterr().out

    assert "Threshold: 0.05000" in output
    assert "value=100.00" in output
    assert "value=150.00" in output
    assert "prediction=" in output
    assert "score=" in output
