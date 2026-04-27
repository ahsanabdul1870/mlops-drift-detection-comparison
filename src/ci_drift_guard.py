"""CI drift guard: run detector checks and emit GitHub Action outputs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mlflow
import pandas as pd

from drift_detector import detect_js, detect_ks, detect_psi
from monitor import update_drift, update_retraining_trigger_time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def _emit_output(name: str, value: str) -> None:
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")


def _load_numeric_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    df = df.replace("?", pd.NA)
    if "income" in df.columns:
        df = df.drop(columns=["income"])
    return df.select_dtypes(include=["number"]).dropna(axis=0)


def main() -> None:
    reference_path = DATA_DIR / "data_clean.csv"
    current_name = os.getenv("CURRENT_DATASET", "data_drift_medium.csv")
    current_path = DATA_DIR / current_name
    drift_level = current_name.replace("data_drift_", "").replace(".csv", "")

    if not reference_path.exists() or not current_path.exists():
        raise FileNotFoundError("Missing drift guard datasets. Run src/drift_simulator.py first.")

    reference = _load_numeric_frame(reference_path)
    current = _load_numeric_frame(current_path)

    mlruns_dir = PROJECT_ROOT / "mlflow" / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    mlflow.set_experiment("adult_income_ci_drift_guard")

    detectors = {"ks": detect_ks, "psi": detect_psi, "js": detect_js}
    summary: dict[str, dict[str, float | bool]] = {}
    detected_detectors: list[str] = []

    for detector_name, detector_fn in detectors.items():
        start = time.perf_counter()
        result = detector_fn(reference, current)
        trigger_seconds = time.perf_counter() - start

        summary[detector_name] = {
            "drift_score": float(result["drift_score"]),
            "drift_detected": bool(result["drift_detected"]),
            "time_taken": float(result["time_taken"]),
            "retraining_trigger_seconds": float(trigger_seconds),
        }

        if bool(result["drift_detected"]):
            detected_detectors.append(detector_name)

        update_drift(
            detector=detector_name,
            drift_score=float(result["drift_score"]),
            drift_detected=bool(result["drift_detected"]),
            drift_level=drift_level,
            time_taken=float(result["time_taken"]),
        )
        update_retraining_trigger_time(detector=detector_name, seconds=float(trigger_seconds))

        with mlflow.start_run(run_name=f"ci_guard_{detector_name}_{drift_level}"):
            mlflow.log_params(
                {
                    "detector_type": detector_name,
                    "drift_level": drift_level,
                    "dataset_version": current_name,
                    "batch_number": "ci",
                }
            )
            mlflow.log_metrics(
                {
                    "drift_score": float(result["drift_score"]),
                    "time_taken": float(result["time_taken"]),
                    "drift_detected": int(bool(result["drift_detected"])),
                    "retraining_trigger_seconds": float(trigger_seconds),
                }
            )
            mlflow.set_tags(
                {
                    "detector_type": detector_name,
                    "drift_level": drift_level,
                    "batch_number": "ci",
                    "pipeline": "github_actions",
                }
            )

    report_path = PROJECT_ROOT / "mlflow" / "monitoring" / "ci_drift_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "drift_detected": bool(detected_detectors),
        "detectors_triggered": detected_detectors,
        "drift_level": drift_level,
        "summary": summary,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _emit_output("drift_detected", "true" if detected_detectors else "false")
    _emit_output("detectors", ",".join(detected_detectors) if detected_detectors else "none")
    _emit_output("primary_detector", detected_detectors[0] if detected_detectors else "none")
    _emit_output("drift_level", drift_level)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
