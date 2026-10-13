
# Sentinel — Anomaly Detection Model

Sentinel is the first specialized model in **ManyMind**, a multi-model machine learning project.

It uses **Isolation Forest**, an unsupervised anomaly detection algorithm from scikit-learn, to identify unusual numerical observations. The project focuses on model training, inference, evaluation, threshold selection, and reproducible testing.

> **Current status:** Sentinel v1 development.
> Evaluation currently uses synthetic data. Real-world performance has not yet been established.

## Features

- Train and persist an Isolation Forest model.
- Load the saved model for inference.
- Classify numerical observations as normal or anomalous.
- Expose anomaly scores through the prediction interface.
- Configure the decision threshold without retraining.
- Evaluate precision, recall, F1, and false-positive rate.
- Run automated tests with pytest.

## Project Structure

```text
ManyMind/
├── manymind/
│   ├── __init__.py
│   └── minds/
│       ├── __init__.py
│       └── sentinel/
│           ├── __init__.py
│           ├── train.py
│           ├── predict.py
│           ├── evaluate.py
│           ├── evaluate_robustness.py
│           ├── calibrate.py
│           ├── compare_thresholds.py
│           └── evaluate_final.py
├── artifacts/
│   └── sentinel/
│       └── model.joblib
├── data/
│   └── sentinel/
├── tests/
├── README.md
└── .gitignore
```

## Requirements

- Python 3.10 or newer
- NumPy
- scikit-learn
- joblib
- pytest for testing

Install the dependencies in your active Python environment:

```bash
python -m pip install numpy scikit-learn joblib pytest
```

Run commands from the repository root so Python can find the `manymind` package.

## Train the Model

```bash
python -m manymind.minds.sentinel.train
```

Training generates normal observations from a synthetic numerical distribution and saves the trained model under `artifacts/sentinel/`.

The current model uses an Isolation Forest with 100 trees and a contamination setting of `0.01`.

The contamination parameter influences the model's default decision boundary; it does not guarantee a real-world anomaly rate.

## Run Inference

Classify one or more values:

```bash
python -m manymind.minds.sentinel.predict 100 115 150
```

Example output:

```text
value=100.00 | prediction=normal | score=0.2652
value=115.00 | prediction=anomaly | score=-0.0700
value=150.00 | prediction=anomaly | score=-0.0731
```

Exact scores depend on the saved model artifact.

### Custom Threshold

Use a more sensitive decision threshold:

```bash
python -m manymind.minds.sentinel.predict --threshold 0.05 100 115 150
```

Sentinel uses the following rule:

- `score < threshold` → anomaly
- `score >= threshold` → normal

A higher threshold flags more observations as anomalous. This can improve recall while increasing false positives.

The default threshold is `0.0`, matching Isolation Forest's existing `decision_function` boundary.

## Evaluation

Sentinel includes scripts for evaluating the model and comparing decision thresholds:

```bash
python -m manymind.minds.sentinel.evaluate
python -m manymind.minds.sentinel.evaluate_robustness
python -m manymind.minds.sentinel.calibrate
python -m manymind.minds.sentinel.compare_thresholds
python -m manymind.minds.sentinel.evaluate_final
```

These experiments use synthetic normal and anomaly distributions.

### Metrics

| Metric | Meaning |
|---|---|
| Precision | Fraction of flagged observations that are actual anomalies |
| Recall | Fraction of actual anomalies detected |
| F1 score | Harmonic mean of precision and recall |
| False-positive rate | Fraction of normal observations incorrectly flagged |

### Threshold Experiments

Initial experiments compared the default threshold (`0.0`) with a candidate threshold (`0.05`).

On one fresh synthetic evaluation:

| Metric | Threshold 0.00 | Threshold 0.05 |
|---|---:|---:|
| Precision | 51.9% | 40.1% |
| Recall | 71.3% | 81.7% |
| F1 score | 60.1% | 53.8% |
| False-positive rate | 1.3% | 2.4% |

These results illustrate the trade-off between catching more anomalies and generating more false alarms. They are results from a controlled synthetic benchmark, not a guarantee of production performance.

The current default remains `0.0`. The `0.05` threshold is an alternative for further evaluation, not a universally optimal setting.

## Run Tests

```bash
python -m pytest -v
```

The tests cover prediction behavior and model training. Run the full suite after modifying model code or threshold logic.

## Limitations

- The current model operates on a single numerical feature.
- Training and evaluation data are synthetic.
- Real-world data may have different distributions and anomaly patterns.
- Thresholds should be selected according to the intended application's tolerance for missed anomalies and false alarms.
- Performance on synthetic data does not establish real-world reliability.

## Roadmap

- [x] Train and persist the first model.
- [x] Implement command-line inference.
- [x] Evaluate robustness across synthetic scenarios.
- [x] Compare decision thresholds.
- [x] Add automated tests.
- [ ] Complete final documentation and repository cleanup.
- [ ] Evaluate with representative real-world data.
- [ ] Integrate Sentinel into the broader ManyMind architecture.

## License

License not yet specified.
