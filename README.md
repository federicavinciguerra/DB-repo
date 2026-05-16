# Databricks MLflow Job Demo

Synthetic regression training project for Databricks Jobs with MLflow tracking.

The repository avoids fragile imports such as `from model import build_model` by using a real Python package:

```text
mlflow-job-demo/
├── src/
│   ├── train.py
│   └── mlflow_job_demo/
│       ├── __init__.py
│       ├── data.py
│       ├── metrics.py
│       ├── models.py
│       └── train.py
├── scripts/
│   └── train.py
├── job_examples/
│   ├── databricks_job_parameters.json
│   └── databricks_bundle_job.example.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

## What it does

The training job:

- Generates synthetic housing-price regression data.
- Splits data into train, validation, and test sets.
- Trains a `GradientBoostingRegressor`.
- Logs CLI/job parameters to MLflow.
- Logs train and validation RMSE across boosting iterations.
- Logs final test RMSE, MAE, and R2.
- Logs the trained scikit-learn model with MLflow signature and input example.
- Optionally logs a CSV sample of test predictions.

## Local run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py \
  --n_estimators 300 \
  --learning_rate 0.03 \
  --max_depth 4 \
  --experiment_name mlflow-job-demo-local
```

## Databricks Job setup from Git provider

Use the Databricks Job task type `Python script`.

Recommended settings:

```text
Source: Git provider
Script path: src/train.py
```

Do not use:

```text
/src/train.py
./src/train.py
```

Task parameters must be separate CLI tokens, for example:

```text
--n_estimators
300
--learning_rate
0.03
--max_depth
4
--experiment_name
/Shared/mlflow-job-demo
```

Or as JSON-style list:

```json
[
  "--n_estimators", "300",
  "--learning_rate", "0.03",
  "--max_depth", "4",
  "--experiment_name", "/Shared/mlflow-job-demo"
]
```

## Databricks job-level parameters

You can define job parameters and pass them to the Python script task with dynamic references.

Example task parameters:

```text
--n_estimators
{{job.parameters.n_estimators}}
--learning_rate
{{job.parameters.learning_rate}}
--max_depth
{{job.parameters.max_depth}}
--experiment_name
{{job.parameters.experiment_name}}
```

See `job_examples/databricks_job_parameters.json` for a complete example.

## Supported CLI parameters

| Parameter | Default | Description |
|---|---:|---|
| `--n_samples` | `5000` | Number of generated rows. |
| `--noise_std` | `20000` | Standard deviation of target noise. |
| `--test_size` | `0.15` | Test fraction of the full dataset. |
| `--val_size` | `0.15` | Validation fraction of the full dataset. |
| `--random_state` | `42` | Seed for data generation and model training. |
| `--n_estimators` | `200` | Number of boosting stages. |
| `--learning_rate` | `0.05` | Gradient boosting learning rate. |
| `--max_depth` | `3` | Max depth of individual regression estimators. |
| `--experiment_name` | env `MLFLOW_EXPERIMENT_NAME` or unset | MLflow experiment name/path. |
| `--run_name` | generated | MLflow run name. |
| `--model_artifact_path` | `model` | MLflow model artifact path. |
| `--registered_model_name` | unset | Optional registry model name. |
| `--metric_log_frequency` | `1` | Log RMSE every N boosting iterations. |
| `--log_sample_predictions` | false | Log sample predictions as a CSV artifact. |
| `--sample_prediction_rows` | `100` | Number of rows in prediction artifact. |

## Recommended experiment runs

Baseline:

```text
--n_estimators 200 --learning_rate 0.05 --max_depth 3
```

Underfitting:

```text
--n_estimators 20 --learning_rate 0.1 --max_depth 1
```

More complex model:

```text
--n_estimators 500 --learning_rate 0.02 --max_depth 5
```

Noisier data:

```text
--noise_std 50000
```

Log sample predictions:

```text
--log_sample_predictions --sample_prediction_rows 50
```

## Notes for Databricks compute

Install dependencies through the Job task environment/libraries or use a Databricks ML Runtime that already includes common ML packages.

`requirements.txt` is included, but creating the file alone does not install the dependencies for a Job.
