
from manymind.minds.sentinel.train import train
from manymind.minds.sentinel.predict import load_model


def test_train_saves_selected_model(capsys):
    train()
    output = capsys.readouterr().out

    assert "Training observations: 1000" in output
    assert "Contamination: 0.01" in output
    assert "Trees: 100" in output
    assert "Model saved to:" in output

    model = load_model()
    assert model.contamination == 0.01
    assert len(model.estimators_) == 100
