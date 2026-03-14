"""
src/features.py
Feature engineering shared by:
  - Training notebook   (Sections 7–8)
  - FastAPI serving     (api/main.py)
  - Prefect retrain     (pipeline/retrain_flow.py)

Keeping all feature logic in one place guarantees training/serving consistency.
Any change here automatically propagates to all three consumers.
"""
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import List

SENSORS               = ["volt", "rotate", "pressure", "vibration"]
WINDOWS               = [3, 6, 12, 24]
LAG_HOURS             = [1, 3, 6]
FAILURE_WINDOW_HOURS  = 48


# ── Label engineering ─────────────────────────────────────────────────────────

def create_failure_labels(
    telemetry_df: pd.DataFrame,
    failures_df:  pd.DataFrame,
    window_hours: int = FAILURE_WINDOW_HOURS,
) -> pd.DataFrame:
    """
    For every (machineID, datetime) row in telemetry_df, set:
      label        = 1  if the machine fails within the next `window_hours` hours
      failure_type = the component that will fail (or "none")
    """
    df     = telemetry_df.copy()
    df["label"]        = 0
    df["failure_type"] = "none"
    window = timedelta(hours=window_hours)

    for machine_id in df["machineID"].unique():
        mach_fail = failures_df[failures_df["machineID"] == machine_id]
        mach_mask = df["machineID"] == machine_id

        for _, fail_row in mach_fail.iterrows():
            fail_time = fail_row["datetime"]
            in_window = (
                mach_mask
                & (df["datetime"] >= fail_time - window)
                & (df["datetime"] <  fail_time)
            )
            df.loc[in_window, "label"]        = 1
            df.loc[in_window, "failure_type"] = fail_row["failure"]

    return df


# ── Core feature builders ─────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean + std for each sensor at 3h / 6h / 12h / 24h windows."""
    df = df.sort_values(["machineID", "datetime"]).copy()
    for sensor in SENSORS:
        for w in WINDOWS:
            grp = df.groupby("machineID")[sensor]
            df[f"{sensor}_roll{w}_mean"] = grp.transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f"{sensor}_roll{w}_std"] = grp.transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0)
            )
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sensor readings 1h / 3h / 6h ago per machine."""
    for sensor in SENSORS:
        for lag in LAG_HOURS:
            col = f"{sensor}_lag{lag}h"
            df[col] = df.groupby("machineID")[sensor].transform(
                lambda x: x.shift(lag)
            )
            # Fill NaN at the start of each machine's history with column median
            df[col] = df[col].fillna(df[col].median())
    return df


def add_placeholder_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    FFT dominant frequency and STL anomaly scores.
    Set to 0 in the serving path (fast inference, no batch context).
    The training notebook overwrites these with real values.
    """
    for sensor in SENSORS:
        if f"{sensor}_fft_dominant" not in df.columns:
            df[f"{sensor}_fft_dominant"]    = 0.0
        if f"{sensor}_prophet_anomaly" not in df.columns:
            df[f"{sensor}_prophet_anomaly"] = 0.0
    return df


# ── Serving path (lightweight — no errors/maint tables needed) ────────────────

def engineer_features_serving(recent_readings: pd.DataFrame) -> pd.DataFrame:
    """
    Fast feature engineering for the API serving path.
    Input:  last 24+ hourly rows for a single machine (raw sensor columns only)
    Output: DataFrame with all feature columns, ready to slice with feature_cols
    """
    df = recent_readings.copy().sort_values("datetime")
    df = add_rolling_features(df)
    df = add_lag_features(df)
    df = add_placeholder_features(df)
    return df


# ── Full training path ────────────────────────────────────────────────────────

def add_error_features(
    master:     pd.DataFrame,
    errors_df:  pd.DataFrame,
) -> tuple[pd.DataFrame, List[str]]:
    """
    Count of each error type per machine in the past 24h.
    Returns (merged_df, error_col_names).
    """
    errors_ohe = pd.get_dummies(errors_df, columns=["errorID"], prefix="error")
    error_cols = [c for c in errors_ohe.columns if c.startswith("error_")]

    errors_ohe = errors_ohe.sort_values(["machineID", "datetime"])
    agg_list   = []

    for mid in errors_ohe["machineID"].unique():
        m_err = (
            errors_ohe[errors_ohe["machineID"] == mid]
            .set_index("datetime")[error_cols]
            .resample("h").sum()
            .rolling(24, min_periods=1).sum()
        )
        m_err["machineID"] = mid
        agg_list.append(m_err.reset_index())

    error_agg = pd.concat(agg_list, ignore_index=True)
    master    = master.merge(error_agg, on=["machineID", "datetime"], how="left")
    master[error_cols] = master[error_cols].fillna(0)

    return master, error_cols


def add_maintenance_features(
    master:   pd.DataFrame,
    maint_df: pd.DataFrame,
) -> tuple[pd.DataFrame, List[str]]:
    """
    Hours since last maintenance per component per machine.
    Returns (merged_df, maint_feature_col_names).
    """
    maint_ohe  = pd.get_dummies(maint_df, columns=["comp"], prefix="maint")
    maint_cols = [c for c in maint_ohe.columns if c.startswith("maint_")]
    ds_list    = []

    for mid in master["machineID"].unique():
        m_maint = (
            maint_ohe[maint_ohe["machineID"] == mid]
            .set_index("datetime").sort_index()
        )
        m_tel = (
            master[master["machineID"] == mid][["datetime"]]
            .set_index("datetime").sort_index()
        )
        for col in maint_cols:
            if col not in m_maint.columns:
                continue
            last_maint   = m_maint[m_maint[col] == 1].index
            hours_since  = []
            for ts in m_tel.index:
                past = last_maint[last_maint <= ts]
                hours_since.append(
                    np.nan if len(past) == 0
                    else (ts - past[-1]).total_seconds() / 3600
                )
            m_tel[f"hrs_since_{col}"] = hours_since

        m_tel["machineID"] = mid
        ds_list.append(m_tel.reset_index())

    days_since_df   = pd.concat(ds_list, ignore_index=True)
    maint_feat_cols = [c for c in days_since_df.columns if c.startswith("hrs_since_")]

    master = master.merge(
        days_since_df[["machineID", "datetime"] + maint_feat_cols],
        on=["machineID", "datetime"],
        how="left",
    )
    # Fill NaN (machine never had maintenance) with a large sentinel value
    fill_val = master[maint_feat_cols].max().max()
    master[maint_feat_cols] = master[maint_feat_cols].fillna(fill_val)

    return master, maint_feat_cols


def build_master_features(
    telemetry: pd.DataFrame,
    failures:  pd.DataFrame,
    errors:    pd.DataFrame,
    maint:     pd.DataFrame,
    machines:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline used by training notebook
    and Prefect retraining flow.

    Steps:
      1. Label engineering (48h failure window)
      2. Machine metadata join
      3. Error frequency features
      4. Hours-since-maintenance features
      5. Rolling window features
      6. Lag features
      7. Placeholder FFT + anomaly columns (overwritten by notebook if needed)
    """
    # 1. Labels
    df = create_failure_labels(telemetry, failures, FAILURE_WINDOW_HOURS)

    # 2. Machine metadata
    df = df.merge(machines, on="machineID", how="left")
    df["model"] = pd.Categorical(df["model"]).codes

    # 3. Error features
    df, _ = add_error_features(df, errors)

    # 4. Maintenance features
    df, _ = add_maintenance_features(df, maint)

    # 5 & 6. Rolling + lag
    df = add_rolling_features(df)
    df = add_lag_features(df)

    # 7. Placeholders (training notebook overwrites with real STL/FFT values)
    df = add_placeholder_features(df)

    return df