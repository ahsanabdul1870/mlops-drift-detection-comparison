"""Retraining entrypoint for CI/CD when drift is detected."""

from __future__ import annotations

import os
import time

from monitor import update_retraining_trigger_time
from train import train_and_log


def main() -> None:
    detector = os.getenv("DETECTOR", "unknown")
    drift_level = os.getenv("DRIFT_LEVEL", "unknown")
    dataset_version = os.getenv("RETRAIN_DATASET", "data_clean.csv")

    start = time.perf_counter()
    metrics = train_and_log(
        dataset_version=dataset_version,
        run_name=f"rf_retrain_{detector}_{drift_level}",
        experiment_name="adult_income_retraining",
        tags={
            "trigger": "drift_detected",
            "detector": detector,
            "drift_level": drift_level,
        },
    )
    trigger_seconds = time.perf_counter() - start

    update_retraining_trigger_time(detector=detector, seconds=trigger_seconds)

    print(f"Retraining completed in {trigger_seconds:.4f}s")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
