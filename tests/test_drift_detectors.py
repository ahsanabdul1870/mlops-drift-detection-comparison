from __future__ import annotations

import pandas as pd

from drift_detector import detect_js, detect_ks, detect_psi


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [10, 10, 11, 11, 12, 12]})
    current = pd.DataFrame({"x": [4, 5, 6, 7, 8, 9], "y": [11, 12, 12, 13, 13, 14]})
    return reference, current


def test_ks_detector_contract() -> None:
    reference, current = _frames()
    result = detect_ks(reference, current)
    assert set(result.keys()) == {"drift_score", "drift_detected", "time_taken"}
    assert result["time_taken"] >= 0


def test_psi_detector_contract() -> None:
    reference, current = _frames()
    result = detect_psi(reference, current)
    assert set(result.keys()) == {"drift_score", "drift_detected", "time_taken"}
    assert result["time_taken"] >= 0


def test_js_detector_contract() -> None:
    reference, current = _frames()
    result = detect_js(reference, current)
    assert set(result.keys()) == {"drift_score", "drift_detected", "time_taken"}
    assert result["time_taken"] >= 0
