from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from mlflow_job_demo.data import create_splits, generate_dataset
from mlflow_job_demo.metrics import compute_metrics, rmse
from mlflow_job_demo.models import build_model


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a synthetic regression model and log the run to MLflow."
    )

    # Data parameters
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--noise_std", type=float, default=20000.0)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--random_state", type=int, default=42)

    # Model parameters
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_depth", type=int, default=3)

    # MLflow parameters
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=os.getenv("MLFLOW_EXPERIMENT_NAME"),
        help="MLflow experiment path/name. Example on Databricks: /Shared/mlflow-job-demo",
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--model_artifact_path", type=str, default="model")
    parser.add_argument(
        "--registered_model_name",
        type=str,
        default=None,
        help="Optional model name for MLflow Model Registry registration.",
    )

    # Logging controls
    parser.add_argument(
        "--metric_log_frequency",
        type=int,
        default=1,
        help="Log train/validation RMSE every N boosting iterations.",
    )
    parser.add_argument(
        "--log_sample_predictions",
        action="store_true",
        help="Log a small CSV artifact with sample predictions.",
    )
    parser.add_argument("--sample_prediction_rows", type=int, default=100)

    return parser.parse_args(argv)


def _run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    return (
        f"gbr_est{args.n_estimators}"
        f"_lr{args.learning_rate}"
        f"_depth{args.max_depth}"
        f"_noise{args.noise_std}"
    )


def _log_cli_params(args: argparse.Namespace) -> None:
    for key, value in vars(args).items():
        if value is not None:
            mlflow.log_param(key, value)


def _validate_args(args: argparse.Namespace) -> None:
    if args.metric_log_frequency <= 0:
        raise ValueError("metric_log_frequency must be greater than 0")
    if args.sample_prediction_rows <= 0:
        raise ValueError("sample_prediction_rows must be greater than 0")


def train(args: argparse.Namespace) -> str:
    """Run training and return the MLflow run ID."""
    _validate_args(args)

    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)

    df = generate_dataset(
        n=args.n_samples,
        noise_std=args.noise_std,
        random_state=args.random_state,
    )

    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(
        df,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    model = build_model(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )

    with mlflow.start_run(run_name=_run_name(args)) as active_run:
        mlflow.set_tags(
            {
                "project": "mlflow-job-demo",
                "model_family": "GradientBoostingRegressor",
                "task_type": "synthetic_regression",
            }
        )
        _log_cli_params(args)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("validation_rows", len(X_val))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("feature_count", X_train.shape[1])

        model.fit(X_train, y_train)

        best_val_rmse = float("inf")
        best_train_rmse = None
        best_iteration = None

        for step, (train_pred, val_pred) in enumerate(
            zip(
                model.staged_predict(X_train),
                model.staged_predict(X_val),
            ),
            start=1,
        ):
            train_rmse = rmse(y_train, train_pred)
            val_rmse = rmse(y_val, val_pred)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_train_rmse = train_rmse
                best_iteration = step

        test_pred = model.predict(X_test)
        test_metrics = compute_metrics(y_test, test_pred)

        mlflow.log_metric("best_val_rmse", best_val_rmse)
        mlflow.log_metric("best_iteration", best_iteration)
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})

        signature = infer_signature(X_train, model.predict(X_train))
        model_kwargs = {
            "sk_model": model,
            "artifact_path": args.model_artifact_path,
            "input_example": X_train.head(5),
            "signature": signature,
        }
        if args.registered_model_name:
            model_kwargs["registered_model_name"] = args.registered_model_name

        mlflow.sklearn.log_model(**model_kwargs)

        if args.log_sample_predictions:
            sample_rows = min(args.sample_prediction_rows, len(X_test))
            prediction_sample = X_test.head(sample_rows).copy()
            prediction_sample["actual_price"] = y_test.head(sample_rows).to_numpy()
            prediction_sample["predicted_price"] = test_pred[:sample_rows]
            prediction_sample["error"] = (
                prediction_sample["predicted_price"] - prediction_sample["actual_price"]
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "sample_predictions.csv"
                prediction_sample.to_csv(output_path, index=False)
                mlflow.log_artifact(str(output_path), artifact_path="predictions")

        print(f"MLflow run ID: {active_run.info.run_id}")
        print(f"Best validation RMSE: {best_val_rmse:.2f} at iteration {best_iteration}")
        print(f"Test RMSE: {test_metrics['rmse']:.2f}")
        print(f"Test MAE: {test_metrics['mae']:.2f}")
        print(f"Test R2: {test_metrics['r2']:.4f}")

        return active_run.info.run_id


def main(argv: Optional[Sequence[str]] = None) -> str:
    args = parse_args(argv)
    return train(args)


if __name__ == "__main__":
    main()
