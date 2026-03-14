"""FastAPI Predictive Maintenance Service
Deploy to Railway: railway up
Test locally: uvicorn api.main:app --reload
"""
import json, pickle, os
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH     = os.getenv('MODEL_PATH',     'models/lgbm_v1.pkl')
FEATURES_PATH  = os.getenv('FEATURES_PATH',  'models/features.json')
REGISTRY_PATH  = os.getenv('REGISTRY_PATH',  'models/model_registry.json')
THRESHOLD_PATH = os.getenv('THRESHOLD_PATH', 'models/threshold.json')
STATUS_PATH    = os.getenv('STATUS_PATH',    'models/status.json')

# ── Global state (populated on startup) ──────────────────────────────────────
state = {}

# ── Lifespan handler (replaces deprecated @app.on_event('startup')) ───────────
# FastAPI deprecated on_event in 0.93. Use lifespan context manager instead.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    with open(MODEL_PATH, 'rb') as f:
        state['model'] = pickle.load(f)
    with open(FEATURES_PATH) as f:
        state['feature_cols'] = json.load(f)
    with open(THRESHOLD_PATH) as f:
        state['threshold'] = json.load(f)['threshold']
    with open(REGISTRY_PATH) as f:
        state['registry'] = json.load(f)
    state['explainer'] = shap.TreeExplainer(state['model'])
    print(f"✅ Model loaded: {state['registry']['production']['version']}")
    yield
    # ── Shutdown (nothing to clean up) ────────────────────────────────────────
    state.clear()

# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title='Predictive Maintenance API',
    description='48h failure probability prediction for industrial machines',
    version='1.0.0',
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class SensorReading(BaseModel):
    machineID:  int   = Field(..., ge=1, le=100)
    datetime:   str                              # ISO: 2016-01-01T00:00:00
    volt:       float = Field(..., ge=0, le=300)
    rotate:     float = Field(..., ge=0, le=600)
    pressure:   float = Field(..., ge=0, le=200)
    vibration:  float = Field(..., ge=0, le=100)

class PredictRequest(BaseModel):
    # Pydantic v2: min_length replaces min_items
    readings: List[SensorReading] = Field(
        ...,
        min_length=24,
        description='At least 24 hourly readings for rolling features'
    )

class ShapFeature(BaseModel):
    feature:    str
    shap_value: float
    direction:  str   # 'increases_risk' | 'decreases_risk'

class PredictResponse(BaseModel):
    machineID:            int
    failure_probability:  float
    risk_level:           str    # LOW / MEDIUM / HIGH / CRITICAL
    will_fail_in_48h:     bool
    top_shap_features:    List[ShapFeature]
    model_version:        str
    timestamp:            str

# ── Feature engineering (mirrors training pipeline) ───────────────────────────
SENSORS   = ['volt', 'rotate', 'pressure', 'vibration']
WINDOWS   = [3, 6, 12, 24]
LAG_HOURS = [1, 3, 6]

def engineer_features(readings_df: pd.DataFrame) -> pd.DataFrame:
    df = readings_df.sort_values('datetime').copy()
    for sensor in SENSORS:
        for w in WINDOWS:
            df[f'{sensor}_roll{w}_mean'] = (
                df[sensor].rolling(w, min_periods=1).mean()
            )
            df[f'{sensor}_roll{w}_std'] = (
                df[sensor].rolling(w, min_periods=1).std().fillna(0)
            )
        for lag in LAG_HOURS:
            df[f'{sensor}_lag{lag}h'] = (
                df[sensor].shift(lag).fillna(df[sensor].median())
            )
        df[f'{sensor}_fft_dominant']    = 0.0
        df[f'{sensor}_prophet_anomaly'] = 0.0
    # Fill any columns present in training but missing here
    for col in state['feature_cols']:
        if col not in df.columns:
            df[col] = 0.0
    return df

def risk_label(prob: float) -> str:
    if prob < 0.25: return 'LOW'
    if prob < 0.50: return 'MEDIUM'
    if prob < 0.75: return 'HIGH'
    return 'CRITICAL'

def top_shap_features(shap_vals: np.ndarray, n: int = 3) -> List[dict]:
    top_idx = np.argsort(np.abs(shap_vals))[-n:][::-1]
    return [
        {
            'feature':    state['feature_cols'][i],
            'shap_value': float(shap_vals[i]),
            'direction':  'increases_risk' if shap_vals[i] > 0 else 'decreases_risk',
        }
        for i in top_idx
    ]

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get('/health')
async def health():
    """Liveness + model version check."""
    status_info = {}
    if Path(STATUS_PATH).exists():
        with open(STATUS_PATH) as f:
            status_info = json.load(f)
    prod = state['registry']['production']
    return {
        'status':          'ok',
        'model_version':   prod['version'],
        'trained_at':      prod['trained_at'],
        'mean_auc':        prod['mean_auc'],
        'drift_detected':  status_info.get('drift_detected', 'unknown'),
        'retrain_needed':  status_info.get('retrain_needed', 'unknown'),
        'timestamp':       datetime.now().isoformat(),
    }

@app.post('/predict', response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Single-machine prediction from a list of hourly sensor readings."""
    # Build DataFrame from request
    readings_df = pd.DataFrame([r.model_dump() for r in request.readings])
    readings_df['datetime'] = pd.to_datetime(readings_df['datetime'])

    # Engineer features, align to training column order
    features_df = engineer_features(readings_df)
    X = features_df[state['feature_cols']].tail(1)

    prob      = float(state['model'].predict_proba(X)[0, 1])
    shap_vals = state['explainer'](X).values[0]

    return PredictResponse(
        machineID=           request.readings[-1].machineID,
        failure_probability= round(prob, 4),
        risk_level=          risk_label(prob),
        will_fail_in_48h=    prob >= state['threshold'],
        top_shap_features=   [ShapFeature(**f) for f in top_shap_features(shap_vals)],
        model_version=       state['registry']['production']['version'],
        timestamp=           datetime.now().isoformat(),
    )

@app.post('/batch_predict')
async def batch_predict(file: UploadFile = File(...)):
    """CSV upload → predictions for every machineID in the file."""
    import io
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content), parse_dates=['datetime'])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Could not parse CSV: {e}')

    required = ['machineID', 'datetime', 'volt', 'rotate', 'pressure', 'vibration']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f'Missing columns: {missing}')

    results = []
    for mid in df['machineID'].unique():
        m_df    = df[df['machineID'] == mid].sort_values('datetime')
        feat_df = engineer_features(m_df)
        X       = feat_df[state['feature_cols']].tail(1)
        prob    = float(state['model'].predict_proba(X)[0, 1])
        results.append({
            'machineID':           int(mid),
            'failure_probability': round(prob, 4),
            'risk_level':          risk_label(prob),
            'will_fail_in_48h':    prob >= state['threshold'],
        })

    results.sort(key=lambda x: x['failure_probability'], reverse=True)
    return {
        'predictions':      results,
        'n_machines':       len(results),
        'high_risk_count':  sum(1 for r in results if r['will_fail_in_48h']),
        'model_version':    state['registry']['production']['version'],
        'timestamp':        datetime.now().isoformat(),
    }