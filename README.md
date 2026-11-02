# ManyMind

ManyMind is an experimental Python machine learning project that organizes specialized models, called **Minds**, behind a shared interface.

It is early-stage. Sentinel and Classifier are trained on **public benchmark datasets**. Those numbers are not a live bank or carrier deployment.

## Current models

Both existing models implement `BaseMind` from `manymind/core/base.py`.

| Mind | Task | Algorithm | Input |
|---|---|---|---|
| **Sentinel** | Credit-card fraud review | Isolation Forest | `V1`–`V28` + `Amount` |
| **Classifier** | Telecom churn | Random Forest pipeline | IBM Telco customer fields |

### Sentinel

Sentinel is an Isolation Forest fraud-review scorer trained on the public ULB credit-card dataset (`V1`–`V28` + `Amount`). A transaction is flagged when `score < threshold`. Training picks the threshold on a calibration split (max F1) and reports holdout ROC-AUC / PR-AUC in `artifacts/sentinel/metrics.json`.

A person uses it by exporting a CSV of transactions, not by typing one number:

```bash
python -m manymind.minds.sentinel.train
python -m manymind.minds.sentinel.predict --csv tests/fixtures/creditcard_sample.csv
```

### Classifier

The classifier is a supervised churn model on the public IBM Telco dataset. The pipeline one-hot encodes categoricals, imputes numerics, then fits a balanced Random Forest. Train and score a customer CSV:

```bash
python -m manymind.minds.classifier.train
python -m manymind.minds.classifier.predict --csv tests/fixtures/telco_sample.csv
```

## Shared BaseMind interface

`manymind/core/base.py` defines the contract:

| Method | Purpose |
|---|---|
| `fit(X, y=None)` | Train the model. `y` is required for the classifier and unused for Sentinel. |
| `predict(X)` | Return model-specific class labels. |
| `score(X)` | Return model-specific scores. These are **not** a shared, comparable scale. |
| `save(path)` | Persist the underlying sklearn object. |
| `load(path)` | Restore a previously saved artifact into the Mind wrapper. |

`score()` is deliberately model-specific:

- **Sentinel** returns Isolation Forest `decision_function` values.
- **Classifier** returns the predicted probability of churn (class `1`).

Do not compare Sentinel scores with classifier probabilities.

### Sentinel scoring

Isolation Forest exposes more than one numeric score. Sentinel uses them as follows:

| Signal | Meaning |
|---|---|
| `score_samples()` | Raw Isolation Forest normality scores. Not exposed by `BaseMind`. |
| `decision_function()` / `Sentinel.score()` | `score_samples` shifted by the fitted offset so the native boundary is `0.0`. Higher values are more normal. |
| Threshold | The CLI labels a value `anomaly` when `score < threshold`. Default threshold: `0.0`. |
| `Sentinel.predict()` | Isolation Forest labels: `1` (normal / inlier) and `-1` (anomaly / outlier), equivalent to `score < 0.0`. |

Custom thresholds are a CLI concern. They are not stored on the `Sentinel` object.

## Architecture

```text
ManyMind/
├── manymind/
│   ├── core/
│   │   └── base.py                 # BaseMind interface
│   └── minds/
│       ├── sentinel/
│       │   ├── model.py            # Sentinel(BaseMind)
│       │   ├── train.py
│       │   ├── predict.py          # CLI inference
│       │   ├── evaluate.py
│       │   ├── evaluate_robustness.py
│       │   ├── calibrate.py
│       │   ├── compare_thresholds.py
│       │   └── evaluate_final.py
│       └── classifier/
│           ├── model.py            # ChurnClassifier(BaseMind)
│           └── train.py
├── artifacts/
│   ├── sentinel/
│   │   ├── model.joblib
│   │   └── threshold.json
│   └── classifier/
│       └── model.joblib
├── data/
│   └── sentinel/                   # generated CSVs (gitignored)
├── tests/
│   ├── test_classifier.py
│   ├── test_sentinel_model.py
│   ├── test_sentinel_predict.py
│   └── test_sentinel_train.py
├── README.md
└── .gitignore
```

`save()` writes the sklearn estimator or pipeline, not the ManyMind wrapper. `load()` wraps that artifact again.

## Setup

There is no `requirements.txt` or `pyproject.toml` yet. Dependency pinning and package installation are future improvements.

Python **3.10+** is required (`str | Path` type syntax). Run commands from the repository root so `python -m manymind...` can import the local package.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn joblib pytest
```

## Datasets

Do not commit the full CSVs. Download them once into gitignored folders.

**Sentinel — ULB credit-card fraud** (OpenML 1597). OpenML checksums are unreliable; use this mirror:

```bash
mkdir -p data/sentinel/creditcard
curl -L --retry 3 -o data/sentinel/creditcard/creditcard.csv \
  "https://huggingface.co/datasets/PELAB-LiU/JunoBench/resolve/main/benchmark/pandas_11/data/creditcard.csv"
```

You want about `(284807, 31)` rows/columns and **492** frauds (`Class == 1`).

**Classifier — IBM Telco Customer Churn** (OpenML 42178):

```bash
mkdir -p data/classifier/telco
curl -L --retry 3 -o data/classifier/telco/telco.csv \
  "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
```

You want about `(7043, 21)` and a `Churn` column (`Yes`/`No`).

Current runtime dependencies, based on imports in the implementation:

| Package | Used for |
|---|---|
| NumPy | Synthetic data and array handling |
| Pandas | Sentinel training CSV output and some evaluation scripts |
| scikit-learn | Isolation Forest, Random Forest pipeline, metrics |
| joblib | Model serialization |
| pytest | Tests |

## Usage

Run these from the repository root, with the virtual environment activated.

Train Sentinel (requires the credit-card CSV):

```bash
python -m manymind.minds.sentinel.train
```

Writes `artifacts/sentinel/model.joblib`, `threshold.json`, and `metrics.json`.

Predict with Sentinel — pass a CSV with columns `V1`…`V28`,`Amount` (shell, not the `>>>` prompt):

```bash
python -m manymind.minds.sentinel.predict --csv tests/fixtures/creditcard_sample.csv
```

Train the classifier (requires the telco CSV):

```bash
python -m manymind.minds.classifier.train
```

Writes `artifacts/classifier/model.joblib` and `metrics.json`.

Predict with the classifier — pass a CSV with IBM Telco feature headers:

```bash
python -m manymind.minds.classifier.predict --csv tests/fixtures/telco_sample.csv
```

Run tests:

```bash
python -m pytest -v
```

The tests cover Sentinel's `BaseMind` behavior, prediction and threshold rules, training artifact output, and classifier fit/predict/save/load. Training tests write to temporary directories and do not replace repository artifacts.

Saved `*.joblib` files may already exist under `artifacts/`. Generated Sentinel CSVs under `data/sentinel/` are gitignored. Do not assume every local artifact is tracked.

## Evaluation and limitations

Sentinel includes experimental scripts that score the saved model on **synthetic** normal and anomaly distributions:

```bash
python -m manymind.minds.sentinel.evaluate
python -m manymind.minds.sentinel.evaluate_robustness
python -m manymind.minds.sentinel.calibrate
python -m manymind.minds.sentinel.compare_thresholds
python -m manymind.minds.sentinel.evaluate_final
```

They report precision, recall, F1, and false-positive rate on generated labels. Those numbers are for threshold exploration. They are not evidence of real-world generalization.

Known limitations:

- All current training and evaluation data are synthetic.
- Sentinel uses a single numeric feature.
- Classifier features and labels are generated from a known synthetic rule, including a small amount of missing values.
- Anomaly thresholds trade precision against recall. A more sensitive threshold catches more anomalies and raises false positives.
- `calibrate.py` can select an alternative threshold on synthetic validation data. That candidate is separate from the CLI default of `0.0`.
- Some evaluation scripts still use a machine-specific project path.
- Saved sklearn artifacts can be sensitive to scikit-learn version differences.
- The repository is experimental and still evolving.

## Development status

**Implemented**

- Shared `BaseMind` interface
- Sentinel Isolation Forest training, CLI inference, and synthetic evaluation scripts
- Classifier Random Forest training on synthetic churn data
- Joblib save/load for both minds
- pytest coverage for Sentinel and the classifier

**Not implemented**

- Formal packaging (`requirements.txt` / `pyproject.toml`)
- Classifier inference CLI
- Portable paths in every Sentinel evaluation script
- Automated CI
- Experiment tracking or model monitoring
- Evaluation on representative real-world data
- License

Possible later work: reproducible evaluation, portable path handling, dependency management, experiment tracking, model monitoring, and CI.

## License

License not yet specified.
