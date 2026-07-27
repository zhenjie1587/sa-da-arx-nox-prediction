"""Data loading, time-delay construction, and sensor-missingness scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from config import (
    CALIBRATION_END,
    CALIBRATION_START,
    DATA_FILE,
    LOAD_PROXY_COLUMN,
    MAX_EXOGENOUS_LAG,
    PROCESS_COLUMNS,
    TARGET_COLUMNS,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
)


@dataclass(frozen=True)
class DatasetBundle:
    raw: pd.DataFrame
    process: pd.DataFrame
    target: pd.DataFrame
    sample_id: pd.Series
    train_idx: np.ndarray
    calibration_idx: np.ndarray
    test_idx: np.ndarray


def load_dataset() -> DatasetBundle:
    """Load the raw ``original`` sheet without modifying any observations."""
    raw = pd.read_excel(DATA_FILE, sheet_name="original", header=None, skiprows=1)
    expected_columns = 1 + len(PROCESS_COLUMNS) + len(TARGET_COLUMNS)
    if raw.shape != (10000, expected_columns):
        raise ValueError(
            f"Expected a 10000 x {expected_columns} original sheet, got {raw.shape}."
        )
    raw.columns = ["sample_id", *PROCESS_COLUMNS, *TARGET_COLUMNS]
    raw = raw.apply(pd.to_numeric, errors="raise")

    if not np.array_equal(raw["sample_id"].to_numpy(), np.arange(1, 10001)):
        raise ValueError("The first column is not the expected 1--10000 sample index.")
    if raw.isna().any().any():
        raise ValueError("Unexpected missing values exist in the source raw sheet.")

    return DatasetBundle(
        raw=raw,
        process=raw.loc[:, PROCESS_COLUMNS].copy(),
        target=raw.loc[:, TARGET_COLUMNS].copy(),
        sample_id=raw["sample_id"].astype(int).copy(),
        train_idx=np.arange(TRAIN_START, TRAIN_END),
        calibration_idx=np.arange(CALIBRATION_START, CALIBRATION_END),
        test_idx=np.arange(TEST_START, TEST_END),
    )


def select_exogenous_lags(
    process: pd.DataFrame,
    target: pd.DataFrame,
    train_idx: np.ndarray,
    max_lag: int = MAX_EXOGENOUS_LAG,
) -> tuple[dict[str, int], pd.DataFrame]:
    """Select one lag per process variable using training-only correlation.

    The selection target is the mean of standardized A- and B-side NOx. This
    prevents the higher-variance A-side channel from dominating lag selection.
    """
    train_target = target.iloc[train_idx]
    standardized_target = (
        (target - train_target.mean(axis=0)) / train_target.std(axis=0, ddof=0)
    ).mean(axis=1)

    rows: list[dict[str, float | int | str]] = []
    lag_map: dict[str, int] = {}
    for column in process.columns:
        correlations: list[float] = []
        for lag in range(max_lag + 1):
            x = process[column].shift(lag).iloc[train_idx].to_numpy(dtype=float)
            y = standardized_target.iloc[train_idx].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            correlation = np.corrcoef(x[valid], y[valid])[0, 1]
            correlations.append(float(abs(correlation)))
        best_lag = int(np.nanargmax(correlations))
        lag_map[column] = best_lag
        rows.append(
            {
                "feature": column,
                "selected_lag_min": best_lag,
                "absolute_correlation": correlations[best_lag],
                "correlation_at_lag_0": correlations[0],
            }
        )
    return lag_map, pd.DataFrame(rows)


def build_features(
    process: pd.DataFrame,
    target: pd.DataFrame,
    lag_map: dict[str, int],
    *,
    include_exogenous_lags: bool,
    include_target_lags: bool,
    target_lags: Iterable[int],
) -> pd.DataFrame:
    """Create process, selected delayed-process, and autoregressive features."""
    features = process.copy()
    if include_exogenous_lags:
        for column, lag in lag_map.items():
            if lag > 0:
                features[f"{column}_lag{lag}"] = process[column].shift(lag)
    if include_target_lags:
        for lag in target_lags:
            for target_column in target.columns:
                features[f"{target_column}_lag{lag}"] = target[target_column].shift(lag)
    return features


def apply_mcar(
    process: pd.DataFrame,
    indices: np.ndarray,
    rate: float,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    """Inject missing completely at random into process sensors only."""
    corrupted = process.copy()
    rng = np.random.default_rng(seed)
    values = corrupted.iloc[indices].to_numpy(dtype=float, copy=True)
    mask = rng.random(values.shape) < rate
    values[mask] = np.nan
    corrupted.iloc[indices] = values
    return corrupted, float(mask.mean())


def apply_block_missing(
    process: pd.DataFrame,
    indices: np.ndarray,
    rate: float,
    seed: int,
    block_length: int = 60,
) -> tuple[pd.DataFrame, float]:
    """Inject contiguous one-hour sensor outages until the target rate is met."""
    corrupted = process.copy()
    rng = np.random.default_rng(seed)
    n_rows = len(indices)
    n_features = process.shape[1]
    mask = np.zeros((n_rows, n_features), dtype=bool)
    target_missing = int(np.ceil(rate * mask.size))
    attempts = 0
    while int(mask.sum()) < target_missing and attempts < target_missing * 3:
        feature_idx = int(rng.integers(0, n_features))
        start = int(rng.integers(0, max(1, n_rows - block_length + 1)))
        stop = min(n_rows, start + block_length)
        mask[start:stop, feature_idx] = True
        attempts += 1
    values = corrupted.iloc[indices].to_numpy(dtype=float, copy=True)
    values[mask] = np.nan
    corrupted.iloc[indices] = values
    return corrupted, float(mask.mean())


def apply_sensor_outage(
    process: pd.DataFrame,
    indices: np.ndarray,
    sensor: str,
) -> tuple[pd.DataFrame, float]:
    """Remove one complete process-sensor stream over an evaluation period."""
    corrupted = process.copy()
    corrupted.loc[indices, sensor] = np.nan
    return corrupted, 1.0 / process.shape[1]


def make_scenario_process(
    process: pd.DataFrame,
    indices: np.ndarray,
    scenario: str,
    replicate: int,
    critical_sensor: str,
) -> tuple[pd.DataFrame, float]:
    """Create one deterministic scenario realization."""
    seed = 10_000 + 101 * replicate
    if scenario == "clean":
        return process.copy(), 0.0
    if scenario == "mcar10":
        return apply_mcar(process, indices, 0.10, seed + 10)
    if scenario == "mcar20":
        return apply_mcar(process, indices, 0.20, seed + 20)
    if scenario == "mcar30":
        return apply_mcar(process, indices, 0.30, seed + 30)
    if scenario == "block20":
        return apply_block_missing(process, indices, 0.20, seed + 40)
    if scenario == "critical_outage":
        return apply_sensor_outage(process, indices, critical_sensor)
    raise KeyError(f"Unknown scenario: {scenario}")


def derive_load_regimes(
    process: pd.DataFrame,
    train_idx: np.ndarray,
) -> tuple[pd.Series, dict[str, float]]:
    """Classify stable, ramp-up, and ramp-down periods using the load proxy."""
    delta = process[LOAD_PROXY_COLUMN].diff()
    threshold = float(delta.iloc[train_idx].abs().quantile(0.75))
    regime = pd.Series("stable", index=process.index, dtype="object")
    regime.loc[delta > threshold] = "ramp_up"
    regime.loc[delta < -threshold] = "ramp_down"

    train_load = process[LOAD_PROXY_COLUMN].iloc[train_idx]
    low_cut, high_cut = train_load.quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
    load_band = pd.cut(
        process[LOAD_PROXY_COLUMN],
        bins=[-np.inf, low_cut, high_cut, np.inf],
        labels=["low_load", "mid_load", "high_load"],
    ).astype("object")
    return pd.DataFrame({"dynamic_regime": regime, "load_band": load_band}), {
        "ramp_threshold_MW_per_min": threshold,
        "low_load_cut_MW": float(low_cut),
        "high_load_cut_MW": float(high_cut),
    }

