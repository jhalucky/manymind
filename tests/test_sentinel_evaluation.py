
import json
from pathlib import Path

import numpy as np
import pytest

from manymind.minds.sentinel import compare_thresholds, evaluate_final
from manymind.minds.sentinel.calibrate import calibrate
from manymind.minds.sentinel.evaluate import evaluate
from manymind.minds.sentinel.evaluate_robustness import evaluate_robustness
from manymind.minds.sentinel.evaluation import (
    CALIBRATION_SEEDS,
    DEFAULT_THRESHOLD,
    HOLDOUT_SEEDS,
    ROBUSTNESS_SEEDS,
    TARGET_RECALL,
    THRESHOLD_CONFIG_PATH,
    TRAIN_SEED,
    ThresholdConfigError,
    assigned_seed_groups,
    calculate_metrics,
    holdout_thresholds,
    labels_from_scores,
    load_calibrated_threshold,
    native_predict_matches_default_threshold,
    resolve_writable_path,
    validate_disjoint_splits,
)
import manymind.minds.sentinel.evaluation as evaluation_module
from manymind.minds.sentinel.model import Sentinel
from manymind.minds.sentinel.predict import MODEL_PATH
from manymind.minds.sentinel.synthetic import (
    LABEL_ANOMALY,
    LABEL_NORMAL,
    generate_labeled_dataset,
    generate_normals,
)
from manymind.minds.sentinel.train import generate_training_data


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ARTIFACT = REPO_ROOT / "artifacts" / "sentinel" / "model.joblib"
THRESHOLD_ARTIFACT = REPO_ROOT / "artifacts" / "sentinel" / "threshold.json"
HISTORICAL_CSVS = [
    REPO_ROOT / "data" / "sentinel" / "final_test.csv",
    REPO_ROOT / "data" / "sentinel" / "evaluation_runs.csv",
    REPO_ROOT / "data" / "sentinel" / "evaluation_summary.csv",
    REPO_ROOT / "data" / "sentinel" / "threshold_comparison.csv",
    REPO_ROOT / "data" / "sentinel" / "final_threshold_evaluation.csv",
]


def _file_fingerprint(path: Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@pytest.fixture
def artifact_fingerprints():
    paths = [MODEL_ARTIFACT, THRESHOLD_ARTIFACT, *HISTORICAL_CSVS]
    before = {path: _file_fingerprint(path) for path in paths}
    yield before
    after = {path: _file_fingerprint(path) for path in paths}
    assert after == before


@pytest.fixture
def fitted_model(tmp_path, monkeypatch, artifact_fingerprints):
    model = Sentinel().fit(generate_normals(seed=TRAIN_SEED, n_normal=200))
    artifact = tmp_path / "model.joblib"
    model.save(artifact)
    monkeypatch.setattr("manymind.minds.sentinel.predict.MODEL_PATH", artifact)
    return model


def test_labeled_generator_is_deterministic():
    first = generate_labeled_dataset(seed=3101, anomaly_mean=120)
    second = generate_labeled_dataset(seed=3101, anomaly_mean=120)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_labeled_generator_uses_documented_label_convention():
    X, y = generate_labeled_dataset(
        seed=1,
        anomaly_mean=150,
        n_normal=40,
        n_anomaly=10,
    )

    assert X.shape == (50, 1)
    assert set(y) == {LABEL_NORMAL, LABEL_ANOMALY}
    assert int(y.sum()) == 10


def test_training_data_are_unlabeled_normals():
    X = generate_training_data()

    assert X.shape == (1000, 1)
    np.testing.assert_array_equal(X, generate_normals(seed=TRAIN_SEED))


def test_split_seeds_are_disjoint():
    validate_disjoint_splits()
    groups = assigned_seed_groups()

    assert TRAIN_SEED not in groups["calibration"]
    assert TRAIN_SEED not in groups["holdout"]
    assert set(CALIBRATION_SEEDS).isdisjoint(HOLDOUT_SEEDS)
    assert set(CALIBRATION_SEEDS).isdisjoint(ROBUSTNESS_SEEDS)
    assert set(HOLDOUT_SEEDS).isdisjoint(ROBUSTNESS_SEEDS)


def test_overlapping_split_assignments_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint_splits(
            {
                "calibration": {3101},
                "holdout": {3101, 9101},
            }
        )


def test_default_threshold_remains_zero():
    assert DEFAULT_THRESHOLD == 0.0
    assert TARGET_RECALL == 0.90


def test_score_below_threshold_is_anomalous():
    scores = np.array([0.2, 0.0, -0.1])
    labels = labels_from_scores(scores, threshold=0.0)

    np.testing.assert_array_equal(labels, np.array([0, 0, 1]))


def test_metrics_handle_zero_division():
    y_true = np.array([0, 0, 0])
    scores = np.array([0.4, 0.3, 0.2])
    metrics = calculate_metrics(y_true, scores, threshold=-1.0)

    assert metrics["tp"] == 0
    assert metrics["fp"] == 0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["false_positive_rate"] == 0.0


def test_native_predict_matches_default_threshold(fitted_model):
    X, _y = generate_labeled_dataset(
        seed=9101,
        anomaly_mean=120,
        n_normal=80,
        n_anomaly=8,
    )

    assert native_predict_matches_default_threshold(fitted_model, X)
    scores = fitted_model.score(X)
    expected = np.where(scores < 0.0, -1, 1)
    np.testing.assert_array_equal(fitted_model.predict(X), expected)


def test_load_calibrated_threshold_reads_without_writing(artifact_fingerprints):
    threshold = load_calibrated_threshold(THRESHOLD_CONFIG_PATH)

    assert threshold == 0.131760116973928
    assert threshold != DEFAULT_THRESHOLD


def test_holdout_thresholds_include_default_and_saved_candidate():
    values = holdout_thresholds(config_path=THRESHOLD_CONFIG_PATH)

    assert values[0] == 0.0
    assert 0.05 in values
    calibrated = load_calibrated_threshold(THRESHOLD_CONFIG_PATH)
    assert calibrated in values
    assert calibrated == 0.131760116973928


def test_missing_required_threshold_file_fails(tmp_path):
    missing = tmp_path / "missing.json"

    with pytest.raises(ThresholdConfigError, match="not found"):
        load_calibrated_threshold(missing, required=True)


def test_missing_optional_threshold_file_returns_none(tmp_path):
    missing = tmp_path / "missing.json"

    assert load_calibrated_threshold(missing, required=False) is None


def test_invalid_json_threshold_config_fails(tmp_path):
    path = tmp_path / "threshold.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ThresholdConfigError, match="valid JSON"):
        load_calibrated_threshold(path)


def test_missing_threshold_key_fails(tmp_path):
    path = tmp_path / "threshold.json"
    path.write_text(json.dumps({"target_recall": 0.9}), encoding="utf-8")

    with pytest.raises(ThresholdConfigError, match="missing the 'threshold' key"):
        load_calibrated_threshold(path)


@pytest.mark.parametrize("value", [None, "nan", "NaN", "inf", "-inf", True, False, [0.1], {}])
def test_invalid_threshold_values_fail(tmp_path, value):
    path = tmp_path / "threshold.json"
    path.write_text(json.dumps({"threshold": value}), encoding="utf-8")

    with pytest.raises(ThresholdConfigError, match="Invalid threshold"):
        load_calibrated_threshold(path)


def test_existing_invalid_optional_config_still_fails(tmp_path):
    path = tmp_path / "threshold.json"
    path.write_text(json.dumps({"threshold": "nan"}), encoding="utf-8")

    with pytest.raises(ThresholdConfigError):
        load_calibrated_threshold(path, required=False)


def test_evaluate_final_requires_valid_candidate(
    fitted_model,
    tmp_path,
):
    with pytest.raises(ThresholdConfigError, match="not found"):
        evaluate_final.main(
            output_path=tmp_path / "holdout.csv",
            config_path=tmp_path / "missing.json",
            seeds=(9101,),
            anomaly_means=(120,),
            n_normal=40,
            n_anomaly=4,
        )


def test_resolve_writable_path_refuses_overwrite(tmp_path):
    existing = tmp_path / "already.csv"
    existing.write_text("keep-me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        resolve_writable_path(existing)

    assert existing.read_text(encoding="utf-8") == "keep-me"


def test_calibrate_refuses_to_overwrite_existing_file(fitted_model, tmp_path):
    existing = tmp_path / "threshold.json"
    existing.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        calibrate(
            output_path=existing,
            seeds=(3101,),
            anomaly_means=(150,),
            target_recall=0.5,
            n_normal=40,
            n_anomaly=5,
        )

    assert existing.read_text(encoding="utf-8") == "{}"


def test_default_outputs_use_run_directory_not_historical_paths(
    fitted_model,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(evaluation_module, "SENTINEL_RUNS_DIR", tmp_path / "runs")

    calibrate(
        seeds=(3101,),
        anomaly_means=(150,),
        target_recall=0.5,
        n_normal=40,
        n_anomaly=5,
    )
    evaluate(data_dir=None)

    run_files = list((tmp_path / "runs").rglob("*"))
    assert any(path.name == "threshold.json" for path in run_files)
    assert any(path.name == "final_test.csv" for path in run_files)
    assert not (REPO_ROOT / "data" / "sentinel" / "final_test.csv").samefile(
        next(path for path in run_files if path.name == "final_test.csv")
    )


def test_calibrate_uses_calibration_seeds_and_tmp_output(fitted_model, tmp_path, capsys):
    output = tmp_path / "threshold.json"

    best = calibrate(
        output_path=output,
        seeds=(3101,),
        anomaly_means=(150,),
        target_recall=0.5,
        n_normal=80,
        n_anomaly=10,
    )

    output_text = capsys.readouterr().out
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output.is_file()
    assert saved["split"] == "calibration"
    assert saved["validation_seeds"] == [3101]
    assert 3101 in CALIBRATION_SEEDS
    assert 3101 not in HOLDOUT_SEEDS
    assert "True positives:" in output_text
    assert "Holdout data were not used" in output_text
    assert best["tp"] + best["fn"] == 10


def test_evaluate_final_scores_holdout_and_saved_threshold(
    fitted_model,
    tmp_path,
    capsys,
):
    config = tmp_path / "threshold.json"
    config.write_text(json.dumps({"threshold": 0.12}), encoding="utf-8")
    output = tmp_path / "holdout.csv"

    evaluate_final.main(
        output_path=output,
        config_path=config,
        seeds=(9101,),
        anomaly_means=(120,),
        n_normal=80,
        n_anomaly=8,
    )

    text = output.read_text(encoding="utf-8")
    printed = capsys.readouterr().out

    assert "split" in text
    assert "holdout" in text
    assert "0.0" in text
    assert "0.05" in text
    assert "0.12" in text
    assert "9101" in printed
    assert "not selected on this data" in printed
    assert set(HOLDOUT_SEEDS).isdisjoint(CALIBRATION_SEEDS)


def test_compare_thresholds_stays_on_calibration_split(
    fitted_model,
    tmp_path,
    capsys,
):
    output = tmp_path / "comparison.csv"

    compare_thresholds.main(
        output_path=output,
        config_path=THRESHOLD_CONFIG_PATH,
        seeds=(3101,),
        anomaly_means=(120,),
        n_normal=80,
        n_anomaly=8,
    )

    printed = capsys.readouterr().out
    assert output.is_file()
    assert "Split: calibration" in printed
    assert "Holdout data were not used" in printed


def test_evaluate_and_robustness_write_to_tmp(
    fitted_model,
    tmp_path,
    capsys,
):
    evaluate(data_dir=tmp_path / "eval")
    evaluate_robustness(
        data_dir=tmp_path / "robust",
        seeds=(2026,),
        anomaly_means=(150,),
        n_normal=80,
        n_anomaly=8,
    )

    assert (tmp_path / "eval" / "final_test.csv").is_file()
    assert (tmp_path / "robust" / "evaluation_runs.csv").is_file()
    assert (tmp_path / "robust" / "evaluation_summary.csv").is_file()
    assert DEFAULT_THRESHOLD == 0.0
    assert MODEL_PATH.name == "model.joblib"
