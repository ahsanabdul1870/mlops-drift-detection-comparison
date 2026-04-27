"""Utilities for runtime monitoring state used by API/Prometheus/Grafana."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _monitoring_dir() -> Path:
    path = _project_root() / "mlflow" / "monitoring"
    path.mkdir(parents=True, exist_ok=True)
    return path


def state_file() -> Path:
    return _monitoring_dir() / "state.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state() -> dict[str, Any]:
    return {
        "last_updated": _now_iso(),
        "accuracy": 0.0,
        "alert_count": 0,
        "latest_drift_level": "clean",
        "drift_scores": {"ks": 0.0, "psi": 0.0, "js": 0.0},
        "drift_detected": {"ks": False, "psi": False, "js": False},
        "detector_time_seconds": {"ks": 0.0, "psi": 0.0, "js": 0.0},
        "retraining_trigger_time_seconds": {"ks": 0.0, "psi": 0.0, "js": 0.0},
    }


def load_monitor_state() -> dict[str, Any]:
    path = state_file()
    if not path.exists():
        return _default_state()
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_monitor_state(state: dict[str, Any]) -> None:
    state["last_updated"] = _now_iso()
    with state_file().open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def update_accuracy(value: float) -> None:
    state = load_monitor_state()
    state["accuracy"] = float(value)
    save_monitor_state(state)


def update_drift(detector: str, drift_score: float, drift_detected: bool, drift_level: str, time_taken: float) -> None:
    state = load_monitor_state()
    state.setdefault("drift_scores", {})[detector] = float(drift_score)
    state.setdefault("drift_detected", {})[detector] = bool(drift_detected)
    state.setdefault("detector_time_seconds", {})[detector] = float(time_taken)
    state["latest_drift_level"] = drift_level
    if drift_detected:
        state["alert_count"] = int(state.get("alert_count", 0)) + 1
    save_monitor_state(state)


def update_retraining_trigger_time(detector: str, seconds: float) -> None:
    state = load_monitor_state()
    state.setdefault("retraining_trigger_time_seconds", {})[detector] = float(seconds)
    save_monitor_state(state)


def monitor_performance() -> None:
    """Print the latest monitoring state for quick terminal checks."""
    state = load_monitor_state()
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    monitor_performance()
