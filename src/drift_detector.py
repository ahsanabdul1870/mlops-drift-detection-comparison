"""Run KS, PSI and JS drift detection experiments and log all runs to MLflow."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, cast

import mlflow
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from monitor import update_drift


TARGET_COLUMN = "income"
EPS = 1e-8


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _numeric_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    df = df.replace("?", np.nan)

    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    return df.select_dtypes(include=[np.number]).dropna(axis=0)


def _common_numeric_columns(reference: pd.DataFrame, current: pd.DataFrame) -> list[str]:
    cols = sorted(set(reference.columns).intersection(current.columns))
    if not cols:
        raise ValueError("No common numeric columns found between reference and current datasets")
    return cols


def detect_ks(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float | bool]:
    """Kolmogorov-Smirnov drift detection over numeric columns."""
    start = time.perf_counter()
    cols = _common_numeric_columns(reference, current)

    statistics = []
    pvalues = []
    for col in cols:
        ks_result = cast(Any, ks_2samp(reference[col].to_numpy(), current[col].to_numpy()))
        statistics.append(float(ks_result.statistic))
        pvalues.append(float(ks_result.pvalue))

    score = float(np.mean(statistics))
    detected = bool(any(p < 0.05 for p in pvalues))
    elapsed = time.perf_counter() - start
    return {"drift_score": score, "drift_detected": detected, "time_taken": elapsed}


def _psi_for_series(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = reference.to_numpy(dtype=float)
    cur = current.to_numpy(dtype=float)

    edges = np.quantile(ref, q=np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        ref_min = float(np.min(ref))
        cur_min = float(np.min(cur))
        ref_max = float(np.max(ref))
        cur_max = float(np.max(cur))
        lo = ref_min if ref_min < cur_min else cur_min
        hi = ref_max if ref_max > cur_max else cur_max
        if lo == hi:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, bins + 1)

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), EPS, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), EPS, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_psi(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float | bool]:
    """Population Stability Index drift detection over numeric columns."""
    start = time.perf_counter()
    cols = _common_numeric_columns(reference, current)

    psi_values = [_psi_for_series(reference[col], current[col]) for col in cols]
    score = float(np.mean(psi_values))
    detected = bool(score > 0.1)
    elapsed = time.perf_counter() - start
    return {"drift_score": score, "drift_detected": detected, "time_taken": elapsed}


def _hist_probs(reference: pd.Series, current: pd.Series, bins: int = 30) -> tuple[np.ndarray, np.ndarray]:
    ref = reference.to_numpy(dtype=float)
    cur = current.to_numpy(dtype=float)

    ref_min = float(np.min(ref))
    cur_min = float(np.min(cur))
    ref_max = float(np.max(ref))
    cur_max = float(np.max(cur))
    lo = ref_min if ref_min < cur_min else cur_min
    hi = ref_max if ref_max > cur_max else cur_max
    if lo == hi:
        hi = lo + 1.0

    edges = np.linspace(lo, hi, bins + 1)
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)

    ref_prob = np.clip(ref_hist / max(ref_hist.sum(), 1), EPS, None)
    cur_prob = np.clip(cur_hist / max(cur_hist.sum(), 1), EPS, None)

    ref_prob = ref_prob / ref_prob.sum()
    cur_prob = cur_prob / cur_prob.sum()
    return ref_prob, cur_prob


def detect_js(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float | bool]:
    """Jensen-Shannon drift detection over numeric columns."""
    start = time.perf_counter()
    cols = _common_numeric_columns(reference, current)

    js_distances = []
    for col in cols:
        ref_prob, cur_prob = _hist_probs(reference[col], current[col])
        js_distances.append(float(jensenshannon(ref_prob, cur_prob)))

    score = float(np.mean(js_distances))
    detected = bool(score > 0.1)
    elapsed = time.perf_counter() - start
    return {"drift_score": score, "drift_detected": detected, "time_taken": elapsed}


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def run_all_experiments() -> None:
    project_root = _project_root()
    data_dir = project_root / "data"
    mlruns_dir = project_root / "mlflow" / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    reference_file = data_dir / "data_clean.csv"
    if not reference_file.exists():
        raise FileNotFoundError(
            "Reference dataset data/data_clean.csv not found. Run src/drift_simulator.py first."
        )

    datasets = {
        "clean": data_dir / "data_clean.csv",
        "light": data_dir / "data_drift_light.csv",
        "medium": data_dir / "data_drift_medium.csv",
        "heavy": data_dir / "data_drift_heavy.csv",
    }

    detectors: dict[str, Callable[[pd.DataFrame, pd.DataFrame], dict[str, float | bool]]] = {
        "ks": detect_ks,
        "psi": detect_psi,
        "js": detect_js,
    }

    reference = _numeric_feature_frame(_load_dataset(reference_file))
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    mlflow.set_experiment("adult_income_drift_detection")

    batch_number = 0
    for detector_name, detector_fn in detectors.items():
        for drift_level, dataset_path in datasets.items():
            batch_number += 1
            current = _numeric_feature_frame(_load_dataset(dataset_path))
            result = detector_fn(reference, current)

            run_name = f"{detector_name}_{drift_level}_batch_{batch_number}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(
                    {
                        "detector_type": detector_name,
                        "drift_level": drift_level,
                        "batch_number": batch_number,
                        "dataset_version": dataset_path.name,
                        "reference_rows": len(reference),
                        "current_rows": len(current),
                    }
                )
                mlflow.log_metrics(
                    {
                        "drift_score": float(result["drift_score"]),
                        "time_taken": float(result["time_taken"]),
                        "drift_detected": int(bool(result["drift_detected"])),
                    }
                )
                mlflow.set_tags(
                    {
                        "detector_type": detector_name,
                        "drift_level": drift_level,
                        "batch_number": str(batch_number),
                    }
                )

            update_drift(
                detector=detector_name,
                drift_score=float(result["drift_score"]),
                drift_detected=bool(result["drift_detected"]),
                drift_level=drift_level,
                time_taken=float(result["time_taken"]),
            )

            print(
                f"{detector_name.upper()} | {drift_level:<6} | "
                f"score={result['drift_score']:.4f} | "
                f"detected={result['drift_detected']} | "
                f"time={result['time_taken']:.4f}s"
            )


if __name__ == "__main__":
    run_all_experiments()
