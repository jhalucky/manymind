
import numpy as np
import pytest

from manymind.minds.classifier.model import ChurnClassifier


@pytest.fixture
def training_data():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(100, 5))
    y = ((X[:, 0] + X[:, 1] * 0.5) > 0).astype(int)

    return X, y


def test_fit_returns_model(training_data):
    X, y = training_data

    model = ChurnClassifier()
    result = model.fit(X, y)

    assert result is model
    assert model.is_fitted


def test_predict_returns_binary_classes(training_data):
    X, y = training_data

    model = ChurnClassifier().fit(X, y)
    predictions = model.predict(X[:10])

    assert len(predictions) == 10
    assert set(predictions).issubset({0, 1})


def test_score_returns_probabilities(training_data):
    X, y = training_data

    model = ChurnClassifier().fit(X, y)
    probabilities = model.score(X[:10])

    assert probabilities.shape == (10,)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_fit_requires_labels(training_data):
    X, _ = training_data

    with pytest.raises(ValueError, match="labels"):
        ChurnClassifier().fit(X)


def test_predict_requires_fitted_model(training_data):
    X, _ = training_data

    with pytest.raises(RuntimeError, match="fitted"):
        ChurnClassifier().predict(X[:5])


def test_save_and_load(tmp_path, training_data):
    X, y = training_data

    model = ChurnClassifier().fit(X, y)
    artifact = tmp_path / "classifier" / "model.joblib"

    model.save(artifact)

    restored = ChurnClassifier().load(artifact)

    np.testing.assert_array_equal(
        model.predict(X[:10]),
        restored.predict(X[:10]),
    )

    np.testing.assert_allclose(
        model.score(X[:10]),
        restored.score(X[:10]),
    )


def test_cannot_save_unfitted_model(tmp_path):
    with pytest.raises(RuntimeError, match="unfitted"):
        ChurnClassifier().save(tmp_path / "model.joblib")
