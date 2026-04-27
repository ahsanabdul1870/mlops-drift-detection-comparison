"""Train the baseline Adult Income model and log benchmark metrics to MLflow."""

from __future__ import annotations

import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from monitor import update_accuracy


TARGET_COLUMN = "income"
RANDOM_SEED = 42


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def _load_clean_dataset(project_root: Path) -> pd.DataFrame:
    data_dir = project_root / "data"
    clean_file = data_dir / "data_clean.csv"
    fallback_file = data_dir / "adult.csv"

    source = clean_file if clean_file.exists() else fallback_file
    if not source.exists():
        raise FileNotFoundError("Could not find clean dataset. Expected data/data_clean.csv or data/adult.csv")

    df = pd.read_csv(source)
    return _prepare_dataframe(df)


def _load_dataset(project_root: Path, dataset_version: str) -> pd.DataFrame:
    dataset_path = project_root / "data" / dataset_version
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    df = pd.read_csv(dataset_path)
    return _prepare_dataframe(df)


def _build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _configure_mlflow(project_root: Path, experiment_name: str) -> None:
    mlruns_dir = project_root / "mlflow" / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    mlflow.set_experiment(experiment_name)


def _save_serving_model(project_root: Path, pipeline: Pipeline) -> None:
    serving_dir = project_root / "mlflow" / "serving_model"
    if serving_dir.exists():
        shutil.rmtree(serving_dir)
    mlflow.sklearn.save_model(sk_model=pipeline, path=str(serving_dir))


def train_and_log(
    dataset_version: str,
    run_name: str,
    experiment_name: str = "adult_income_baseline",
    tags: dict[str, str] | None = None,
) -> dict[str, float]:
    project_root = _project_root()
    _configure_mlflow(project_root, experiment_name)

    df = _load_dataset(project_root, dataset_version)
    df = df.replace("?", np.nan).dropna(axis=0)
    y = df[TARGET_COLUMN].astype(str).str.contains(">50K").astype(int)
    X = df.drop(columns=[TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    pipeline = _build_pipeline(X)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": 300,
                "max_depth": 12,
                "min_samples_leaf": 2,
                "random_state": RANDOM_SEED,
                "dataset_version": dataset_version,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            }
        )
        mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    _save_serving_model(project_root, pipeline)
    update_accuracy(metrics["accuracy"])
    return metrics


def main() -> None:
    project_root = _project_root()
    clean_dataset = "data_clean.csv" if (project_root / "data" / "data_clean.csv").exists() else "adult.csv"
    metrics = train_and_log(
        dataset_version=clean_dataset,
        run_name="rf_baseline_clean",
        experiment_name="adult_income_baseline",
        tags={"benchmark": "pre_drift"},
    )

    print("Baseline run logged to MLflow with pre-drift benchmark metrics.")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
