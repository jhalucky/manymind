
from manymind.minds.sentinel.model import Sentinel
from manymind.minds.sentinel.train import train


def test_train_saves_selected_model(tmp_path, capsys):
    data_dir = tmp_path / "data"
    artifact_dir = tmp_path / "artifacts"

    train(data_dir=data_dir, artifact_dir=artifact_dir)
    output = capsys.readouterr().out

    assert "Training observations: 1000" in output
    assert "Contamination: 0.01" in output
    assert "Trees: 100" in output
    assert "Model saved to:" in output

    model_path = artifact_dir / "model.joblib"
    assert model_path.is_file()
    assert (data_dir / "train.csv").is_file()

    model = Sentinel().load(model_path)
    assert model.contamination == 0.01
    assert len(model.estimator.estimators_) == 100
