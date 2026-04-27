"""Generate synthetic drift variants for the Adult Income dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
TARGET_COLUMN = "income"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_source_dataset(data_dir: Path) -> pd.DataFrame:
    source_file = data_dir / "adult.csv"
    if not source_file.exists():
        raise FileNotFoundError(f"Expected dataset file not found: {source_file}")

    df = pd.read_csv(source_file)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def _shift_numeric(df: pd.DataFrame, rows: np.ndarray, shift_ratio: float, rng: np.random.Generator) -> None:
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET_COLUMN]
    for col in numeric_cols:
        std = df[col].std(ddof=0)
        if np.isnan(std) or std == 0:
            continue
        direction = 1.0 if rng.random() > 0.5 else -1.0
        offset = direction * std * (1.5 * shift_ratio)
        noise = rng.normal(loc=0.0, scale=std * 0.1, size=len(rows))
        shifted = df.loc[rows, col].astype(float) + offset + noise

        if np.issubdtype(df[col].dtype, np.integer):
            shifted = np.round(shifted).astype(int)
            shifted = np.maximum(shifted, 0)
        df.loc[rows, col] = shifted


def _shift_categorical(df: pd.DataFrame, rows: np.ndarray, shift_ratio: float, rng: np.random.Generator) -> None:
    categorical_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != TARGET_COLUMN]
    for col in categorical_cols:
        categories = [c for c in df[col].dropna().unique().tolist() if c != "?"]
        if len(categories) < 2:
            continue

        change_mask = rng.random(len(rows)) < min(1.0, 0.5 + shift_ratio)
        for idx, should_change in zip(rows, change_mask):
            if not should_change:
                continue
            current_value = df.at[idx, col]
            alternatives = [c for c in categories if c != current_value]
            if not alternatives:
                continue
            df.at[idx, col] = rng.choice(alternatives)


def apply_distribution_shift(df: pd.DataFrame, shift_ratio: float, seed: int) -> pd.DataFrame:
    """Apply synthetic feature distribution shift to a fraction of rows."""
    rng = np.random.default_rng(seed)
    shifted = df.copy(deep=True)

    row_count = len(shifted)
    rows_to_shift = int(row_count * shift_ratio)
    if rows_to_shift <= 0:
        return shifted

    selected_rows = rng.choice(shifted.index.to_numpy(), size=rows_to_shift, replace=False)
    _shift_numeric(shifted, selected_rows, shift_ratio, rng)
    _shift_categorical(shifted, selected_rows, shift_ratio, rng)
    return shifted


def main() -> None:
    data_dir = _project_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = _load_source_dataset(data_dir)
    clean_file = data_dir / "data_clean.csv"
    source.to_csv(clean_file, index=False)

    drift_specs = {
        "data_drift_light.csv": 0.10,
        "data_drift_medium.csv": 0.30,
        "data_drift_heavy.csv": 0.60,
    }

    for i, (filename, ratio) in enumerate(drift_specs.items(), start=1):
        drifted = apply_distribution_shift(source, shift_ratio=ratio, seed=RANDOM_SEED + i)
        drifted.to_csv(data_dir / filename, index=False)

    print("Created drift datasets:")
    print(f"- {clean_file.name}")
    for filename in drift_specs:
        print(f"- {filename}")


if __name__ == "__main__":
    main()
