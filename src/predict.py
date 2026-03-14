"""
src/predict.py
Thin inference wrapper reused by FastAPI and any ad-hoc batch scripts.
"""
import json, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from src.features import engineer_features_serving, SENSORS


def load_artifacts(
    model_path:    str = "models/lgbm_v1.pkl",
    features_path: str = "models/features.json",
    threshold_path:str = "models/threshold.json",
) -> tuple:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    feature_cols = json.loads(Path(features_path).read_text())
    threshold    = json.loads(Path(threshold_path).read_text())["threshold"]
    return model, feature_cols, threshold


def predict_single(
    recent_df:    pd.DataFrame,
    model,
    feature_cols: list,
    threshold:    float,
) -> dict:
    """
    Predict failure probability for one machine from its last 24+ hourly rows.

    Returns dict with:
      failure_probability, risk_level, will_fail_in_48h
    """
    feat_df = engineer_features_serving(recent_df)

    # Align to training column set — fill any missing cols with 0
    X = feat_df.reindex(columns=feature_cols, fill_value=0).tail(1)

    prob = float(model.predict_proba(X)[0, 1])

    risk = (
        "CRITICAL" if prob >= 0.75 else
        "HIGH"     if prob >= 0.50 else
        "MEDIUM"   if prob >= 0.25 else "LOW"
    )

    return {
        "failure_probability": round(prob, 4),
        "risk_level":          risk,
        "will_fail_in_48h":    prob >= threshold,
    }