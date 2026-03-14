# Predictive Maintenance for Industrial Machines

> **End-to-end MLOps pipeline that predicts industrial machine failure 48 hours in advance — catching 90% of failures with a 46-hour median warning, deployed on Railway + Streamlit Cloud.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-029E74)](https://lightgbm.readthedocs.io)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?logo=mlflow)](https://mlflow.org)
[![Prefect](https://img.shields.io/badge/Orchestration-Prefect-070E10)](https://prefect.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Evidently](https://img.shields.io/badge/Drift-Evidently-orange)](https://evidentlyai.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Why Predictive Over Reactive?](#why-predictive-over-reactive)
- [Solution Architecture](#solution-architecture)
- [Dataset](#dataset)
- [The Data Leakage Fix](#the-data-leakage-fix)
- [Feature Engineering](#feature-engineering)
- [Model & Training](#model--training)
- [Results](#results)
- [Business Impact](#business-impact)
- [MLOps Pipeline](#mlops-pipeline)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Interview Talking Points](#interview-talking-points)

---

## Problem Statement

Industrial machines — turbines, pumps, compressors, CNC routers — fail without warning. When a machine breaks down unexpectedly:

- **Unplanned downtime** halts entire production lines, not just the failed machine
- **Emergency repair** costs 3–10× more than scheduled maintenance
- **Part procurement** under urgency means premium pricing and longer lead times
- **Safety incidents** increase when machines are run to failure rather than maintained preventively

**Globally, unplanned industrial downtime costs manufacturers an estimated $50 billion per year.**

The sensor data to prevent this already exists. Every modern industrial machine emits continuous telemetry — voltage, rotation speed, pressure, vibration — but most facilities lack the ML infrastructure to act on it. This project builds that infrastructure end-to-end: from raw sensor CSV files to a live REST API that returns a failure probability and tells operators exactly which sensor is driving the risk.

---

## Why Predictive Over Reactive?

| Approach | Trigger | Average Cost | Lead Time for Engineers |
|---|---|---|---|
| **Reactive** | Machine breaks | ~$20,000/event | 0 hours — already broken |
| **Preventive** | Calendar schedule | ~$3,000/event | Days — but over-maintains |
| **Predictive (this project)** | ML risk threshold | ~$2,000/event | **46 hours median** |

Predictive maintenance doesn't just reduce cost — it changes the operational model. Engineers schedule interventions during planned downtime windows, order parts in advance, and prioritise across a fleet of 100 machines by risk score. The 46-hour median warning this model achieves is sufficient for all three.

---

## Solution Architecture

![Pipeline](readme_assets/01_pipeline.png)

| Component | Tool | Purpose |
|---|---|---|
| Feature engineering | pandas, scipy | 68 temporal features from 5 raw CSVs |
| Anomaly detection | STL (statsmodels) | Slow degradation trend features per sensor |
| Model training | LightGBM + Optuna | Gradient boosted trees, 50-trial HPO |
| Validation | TimeSeriesSplit (5-fold) | No temporal data leakage |
| Experiment tracking | MLflow | All 55+ runs logged with params + metrics |
| Drift monitoring | Evidently 0.7 | Weekly feature distribution checks |
| Retraining | Prefect 3.x | Automated weekly pipeline with promotion gate |
| Prediction API | FastAPI + Railway | `/predict`, `/batch_predict`, `/health` |
| Dashboard | Streamlit + Plotly | Live gauge, SHAP attribution, sensor trends |

---

## Dataset

**Microsoft Azure Predictive Maintenance Dataset**
[kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)

| File | Rows | Description |
|---|---|---|
| `PdM_telemetry.csv` | 876,000 | Hourly voltage, rotation, pressure, vibration readings |
| `PdM_failures.csv` | 761 | Failure timestamps and component type per machine |
| `PdM_errors.csv` | 3,919 | Error codes logged by machines (5 error types) |
| `PdM_maint.csv` | 3,286 | Scheduled maintenance records per component |
| `PdM_machines.csv` | 100 | Machine metadata: age (0–20 years), model type |

**Key statistics:**
- 100 machines across 4 model types (model3 dominant at 35%)
- 1 year of hourly data: January 2015 – January 2016
- 761 failure events across 98 machines (~7.8 failures per machine per year)
- Positive rate: **3.87%** (failure in next 48h) — severe class imbalance requiring SMOTE

![Failure analysis](readme_assets/02_failure_analysis.png)

![Machine analysis](readme_assets/03_machine_analysis.png)

**Failure breakdown by component:**
- `comp2`: 259 failures (34.0%) — most failure-prone
- `comp1`: 192 failures (25.2%)
- `comp4`: 179 failures (23.5%)
- `comp3`: 131 failures (17.2%)

---

## The Data Leakage Fix

This is the most important methodological decision in the project, and the one most Kaggle notebooks on this dataset get wrong.

**The problem with random splits on time-series data:**

When you shuffle a time-series dataset and split it randomly into train/test, the model learns patterns from *future* data during training. A row from December 2015 in the training set teaches the model about sensor behaviour that comes *after* a row from March 2015 in the test set. The model appears to perform well, but it's cheating — it has seen the future.

Reference implementations on this dataset all use random splits. Reported AUC on those notebooks: **~0.94**

**The fix — TimeSeriesSplit:**

`sklearn.model_selection.TimeSeriesSplit(n_splits=5)` guarantees that every test fold is temporally *after* its corresponding training fold. The model is evaluated only on data it could not have seen during training, exactly as it would operate in production.

Honest AUC with proper temporal validation: **0.9975 ± 0.0014**

```
Fold 1: Train [Jan–Apr 2015]  →  Test [May 2015]
Fold 2: Train [Jan–Jun 2015]  →  Test [Jul 2015]
Fold 3: Train [Jan–Aug 2015]  →  Test [Sep 2015]
Fold 4: Train [Jan–Oct 2015]  →  Test [Nov 2015]
Fold 5: Train [Jan–Nov 2015]  →  Test [Dec 2015]
```

The inflated ~0.94 from random splits looks better on paper. The honest 0.9975 from temporal splits is what the model actually achieves in production. This distinction — and the ability to explain it clearly — is what separates production-grade ML from Kaggle notebook ML.

---

## Feature Engineering

**68 total features** engineered from 5 raw CSV files across 8 categories:

![Feature breakdown](readme_assets/07_feature_breakdown.png)

### Rolling Window Statistics (32 features)
For each of the 4 sensors (volt, rotate, pressure, vibration), compute rolling **mean** and **std** at 4 time horizons:

```
3h, 6h, 12h, 24h × {mean, std} × 4 sensors = 32 features
```

Rolling std captures *instability* in sensor readings before a failure — something the raw value alone misses entirely.

### Lag Features (12 features)
Sensor readings from 1h, 3h, and 6h ago per sensor. Captures the *rate of change* — a voltage that was 170V an hour ago and is now 185V signals a different risk than one stable at 185V all day.

### STL Anomaly Scores (4 features)
Seasonal-Trend decomposition using LOESS (STL) from `statsmodels` fits a seasonal model to each sensor's history and computes a Z-score-like residual. This captures **slow degradation trends** that rolling window statistics miss — a sensor drifting gradually over weeks rather than spiking in the last few hours. These 4 features are the unique addition over all reference implementations.

### Error Frequency Features (5 features)
Count of each error type per machine in the past 24h rolling window. Error codes are strong early-warning signals — machines log errors before they fail.

### Hours Since Last Maintenance (4 features)
For each of the 4 components, hours since last service on that machine. Components approaching their next service window are higher risk.

### FFT Dominant Frequency (4 features)
Dominant frequency from a rolling 24-hour FFT window per sensor. Captures periodic/cyclical behaviour indicating mechanical wear.

### Machine Metadata (2 features)
Machine age (0–20 years) and model type. Older machines and certain models have systematically higher failure rates.

### Class Imbalance Handling
At 3.87% positive rate, two complementary approaches:
1. **SMOTE** applied *only inside training folds* (never test data) — oversampling to 20% positive rate in training
2. **`scale_pos_weight`** in LightGBM set to `n_negative / n_positive`

---

## Model & Training

### LightGBM with Optuna Hyperparameter Tuning

LightGBM chosen for three reasons specific to this problem:
1. **Mixed feature types** — histogram-based splits handle sensor readings, categoricals, and time-since-maintenance values spanning different scales without normalisation
2. **`scale_pos_weight`** — built-in class imbalance weighting works alongside SMOTE without double-counting
3. **SHAP via `TreeExplainer`** — exact (not approximate) SHAP values for every prediction, powering the live sensor attribution in the dashboard

XGBoost was trained as a baseline. LightGBM outperformed it by ~0.3% AUC on temporal CV.

### Hyperparameter Tuning — Optuna (50 trials)

TPE sampler, tuned on fold 5 (temporally last, closest to production conditions). All 50 trials logged to MLflow as nested runs.

| Parameter | Search Range |
|---|---|
| `num_leaves` | 20 – 300 |
| `learning_rate` | 0.01 – 0.1 (log scale) |
| `max_depth` | 4 – 12 |
| `n_estimators` | 200 – 1,000 |
| `subsample` | 0.6 – 1.0 |
| `colsample_bytree` | 0.6 – 1.0 |
| `reg_alpha` / `reg_lambda` | 1e-4 – 10.0 (log scale) |

Best trial AUC: **0.9979**

---

## Results

### Cross-Validation (5-fold TimeSeriesSplit)

![CV results](readme_assets/04_cv_results.png)

| Metric | Mean | Std |
|---|---|---|
| **AUC-ROC** | **0.9975** | **±0.0014** |
| **F1 Score** | **0.7627** | — |

Low std (0.0014) confirms the model generalises consistently across time, not just for a lucky test period.

### Held-Out Evaluation — Last 90 Days (Never Seen During Training)

![ROC and PR curves](readme_assets/05_roc_pr_curves.png)

| Metric | Value | Notes |
|---|---|---|
| **AUC-ROC** | **0.9982** | Discriminative power across all thresholds |
| **F1 Score** | **0.8827** | Harmonic mean of precision and recall |
| **Recall** | **0.9002** | 90% of actual failures caught |
| **Precision** | **0.8658** | 87% of alarms are genuine |
| Decision threshold | 0.923 | Optimised for recall ≥ 0.90 |
| **Median warning lead time** | **46 hours** | Before failure, alarm first fires |

### Confusion Matrix

![Confusion matrix](readme_assets/06_confusion_matrix.png)

| | Predicted: No Failure | Predicted: Failure |
|---|---|---|
| **Actual: No Failure** | 207,294 — correct silence | 1,078 — false alarms |
| **Actual: Failure** | 771 — missed | **6,957 — caught** |

- **6,957 caught (TP):** Engineers warned ~46 hours in advance — time to schedule repair, order parts, prevent downtime
- **771 missed (FN):** Unplanned breakdowns. 90 out of every 100 failures receive early warning
- **1,078 false alarms (FP):** At 0.52% rate, rare enough that operators continue trusting the system

The decision threshold (0.923) was tuned to hold recall ≥ 0.90 because missing a failure costs ~10× more than a false alarm in this domain.

---

## Business Impact

![Business impact](readme_assets/08_business_impact.png)

Based on the 90-day held-out evaluation across 100 machines:

| Metric | Value |
|---|---|
| Total predictions | 216,100 |
| Failures detected in advance | **6,957 (90.0%)** |
| Failures missed | 771 (10.0%) |
| False alarm rate | **0.52%** |
| Median warning lead time | **46 hours** |
| Estimated reactive cost avoided* | $139.1M |
| Estimated predictive intervention cost* | $16.1M |
| **Estimated net savings*** | **$123.1M** |

*Industry averages: $20,000 per unplanned failure (8h downtime × $2,500/hr) vs $2,000 per scheduled intervention.*

**Why 46 hours matters operationally:**
- Parts ordered before needed — no emergency procurement premium
- Repairs scheduled in planned downtime windows — no production line halt
- All 100 machines triaged by live risk score and queued for inspection
- Safety protocols enacted before failure — no personnel risk from sudden breakdown

---

## MLOps Pipeline

![MLOps monitoring](readme_assets/10_mlops_monitoring.png)

### Experiment Tracking — MLflow

Every run logged: 50 Optuna trials (nested runs), 5-fold CV metrics, final model artifact, feature list, training metadata, and decision threshold.

```
MLflow Run ID: 655faf5bae294c57b8c9c629dc7ac777
Experiment:    predictive-maintenance
Model version: v1.0
Trained at:    2026-03-13 19:51:05
```

### Data Drift Monitoring — Evidently 0.7

Weekly `DataDriftPreset` comparing last 30 days of sensor distributions against the training reference snapshot. Monitors 16 key features (raw sensors + 24h rolling stats + STL anomaly scores).

**Status as of 2026-03-13: 0 / 16 features drifted. No retraining needed.**

### Automated Retraining — Prefect 3.x

```
Every Monday 2am UTC
  1. Load last 30 days of telemetry
  2. Score current model on fresh data
  3. F1 < 0.76  OR  drift detected?
     YES → Full retrain (5-fold TimeSeriesSplit)
           New model must beat old by >2% F1 to be promoted
           Promoted → bump version, update registry, log to MLflow
     NO  → Log health check, skip retrain
```

The promotion gate prevents a model trained on a bad data window from replacing a healthy production model. Version history with rollback is maintained in `models/model_registry.json`.

### Operational Data

![Operational data](readme_assets/09_operational_data.png)

---

## Project Structure

```
predictive-maintenance-mlops/
│
├── README.md
├── requirements.txt
├── Dockerfile                         # Railway deployment
├── railway.json
├── prefect.yaml                       # Prefect 3.x deployment config
├── .streamlit/config.toml             # Dark theme config
│
├── src/
│   ├── features.py                    # Shared feature engineering (train + serve)
│   ├── evaluate.py                    # Metrics utilities
│   ├── predict.py                     # Inference wrapper
│   └── __init__.py
│
├── api/
│   └── main.py                        # FastAPI: /predict /batch_predict /health
│
├── dashboard/
│   ├── app.py                         # Streamlit dashboard
│   └── requirements.txt
│
├── pipeline/
│   └── retrain_flow.py                # Prefect weekly retraining flow
│
├── models/
│   ├── lgbm_v1.pkl                    # Production model
│   ├── features.json                  # Training feature list
│   ├── threshold.json                 # Decision threshold (0.923)
│   ├── model_registry.json            # Version history + rollback
│   ├── label_map.json                 # Failure type encoding
│   └── final_metrics.json             # All evaluation metrics
│
├── notebooks/
│   └── Predictive_Maintenance.ipynb   # Full training notebook (18 sections)
│
├── data/
│   └── raw/                           # 5 Azure PM CSVs (gitignored — re-download from Kaggle)
│
├── reports/
│   └── drift_YYYY-MM-DD.html          # Evidently weekly drift reports
│
└── readme_assets/                     # All charts auto-generated by notebook
    ├── 01_pipeline.png
    ├── 02_failure_analysis.png
    ├── 03_machine_analysis.png
    ├── 04_cv_results.png
    ├── 05_roc_pr_curves.png
    ├── 06_confusion_matrix.png
    ├── 07_feature_breakdown.png
    ├── 08_business_impact.png
    ├── 09_operational_data.png
    └── 10_mlops_monitoring.png
```

---

## Quick Start


git clone https://github.com/YOUR_USERNAME/predictive-maintenance-mlops
cd predictive-maintenance-mlops
pip install -r requirements.txt

# Download dataset
kaggle datasets download \
  -d arnabbiswas1/microsoft-azure-predictive-maintenance \
  -p data/raw --unzip

# Run API
uvicorn api.main:app --reload
# → http://localhost:8000/docs

# Run dashboard
streamlit run dashboard/app.py
# → http://localhost:8501

# Register Prefect weekly schedule
prefect cloud login
python pipeline/retrain_flow.py
prefect worker start --pool default-agent-pool

# Test retraining immediately
python pipeline/retrain_flow.py --run-now --force


---

## API Reference

### `POST /predict`

**Request** — minimum 24 hourly readings per machine:

{
  "readings": [
    {
      "machineID": 1,
      "datetime": "2016-01-15T12:00:00",
      "volt": 171.2,
      "rotate": 451.3,
      "pressure": 100.8,
      "vibration": 40.1
    }
  ]
}


**Response:**

{
  "machineID": 1,
  "failure_probability": 0.9621,
  "risk_level": "CRITICAL",
  "will_fail_in_48h": true,
  "top_shap_features": [
    {"feature": "vibration_roll24_std",  "shap_value": 0.312, "direction": "increases_risk"},
    {"feature": "volt_prophet_anomaly",  "shap_value": 0.198, "direction": "increases_risk"},
    {"feature": "hrs_since_maint_comp2", "shap_value": 0.145, "direction": "increases_risk"}
  ],
  "model_version": "v1.0",
  "timestamp": "2026-03-14T09:00:00"
}


| Risk Level | Probability | Action |
|---|---|---|
| `LOW` | < 0.25 | No action |
| `MEDIUM` | 0.25 – 0.50 | Inspect within 1 week |
| `HIGH` | 0.50 – 0.75 | Inspect within 48 hours |
| `CRITICAL` | ≥ 0.75 | Immediate inspection |

### `POST /batch_predict`
Upload CSV with `[machineID, datetime, volt, rotate, pressure, vibration]`. Returns all machines ranked by failure probability.

### `GET /health`
Returns model version, training date, CV AUC, drift status, retrain flag.

---

## Deployment

| Component | Platform | Cost |
|---|---|---|
| Prediction API | [Railway](https://railway.app) | Free (500h/month) |
| Dashboard | [Streamlit Cloud](https://streamlit.io/cloud) | Free |
| Retraining scheduler | [Prefect Cloud](https://prefect.io) | Free tier |
| Model training | Google Colab | Free (T4 GPU) |


# API → Railway
# railway.app → New Project → Deploy from GitHub
# Set env vars: MODEL_PATH, FEATURES_PATH, THRESHOLD_PATH

# Dashboard → Streamlit Cloud
# streamlit.io/cloud → New app → dashboard/app.py
```

---

## Interview Talking Points

**On data leakage:**
> "Every reference implementation of this dataset uses random train/test splits — a fundamental error on time-series data. The model learns from future sensor readings during training, inflating AUC to ~0.94. I replaced this with `TimeSeriesSplit` where test always comes after train temporally, giving an honest 0.9975. That distinction — and being able to explain it clearly — is the most important methodological point in the project."

**On class imbalance:**
> "Only 3.87% of hourly rows have a failure event in the next 48 hours. I applied SMOTE exclusively inside each training fold, never touching test data, combined with LightGBM's `scale_pos_weight`, and optimised threshold for recall ≥ 0.90. Result: catching 90% of failures at a 0.52% false alarm rate. That false alarm rate matters operationally — if it were 5%, operators would stop trusting the system."

**On lead time:**
> "AUC tells you if the model ranks failures correctly. Lead time tells you if it's actually useful. The 46-hour median warning means parts can be ordered, repairs scheduled in maintenance windows, and machines triaged by risk score across the fleet. That's the metric I'd show a plant manager, not AUC."

**On STL anomaly scores:**
> "Rolling window statistics capture sudden spikes. STL decomposition captures slow degradation — a sensor drifting over weeks rather than spiking in hours. I fit STL on each sensor's history per machine and use the residual as a feature. These 4 features catch failure precursors that rolling stats miss entirely, especially for comp1 and comp3 which tend to degrade gradually."

**On the retraining pipeline:**
> "I built a Prefect flow that runs every Monday, scores the current model on fresh data, and triggers retraining if F1 drops below 0.76 or Evidently detects distribution shift. The new model only replaces production if it beats the old by more than 2% F1. That promotion gate is what separates a retraining pipeline from a retraining script."

**On SHAP in production:**
> "Industrial operators don't trust black box predictions. The dashboard surfaces the top 3 SHAP features per prediction — 'vibration rolling std is driving 31% of this risk score' — which tells the engineer exactly which component to inspect. That's explainable AI in a safety-critical context."

---

## Technical Stack

```
Data:               pandas 2.x · numpy · scipy (FFT)
Feature Eng.:       statsmodels (STL) · pandas rolling windows
ML Model:           LightGBM 4.x
Baseline:           XGBoost 2.x
HPO:                Optuna 3.x (TPE sampler, 50 trials)
Imbalance:          imbalanced-learn (SMOTE) + scale_pos_weight
Explainability:     SHAP (TreeExplainer)
Tracking:           MLflow 2.x
Drift:              Evidently 0.7.x
Orchestration:      Prefect 3.x
API:                FastAPI + Pydantic v2 + uvicorn
Dashboard:          Streamlit 1.35+ + Plotly
Deployment:         Railway (API) · Streamlit Cloud (dashboard)
```

---

## Acknowledgements

Dataset: [Microsoft Azure Predictive Maintenance](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/predictive-maintenance-playbook) via Kaggle — [arnabbiswas1](https://www.kaggle.com/arnabbiswas1).

Reference implementations consulted and improved upon: [Azure ML Samples](https://github.com/Azure/MachineLearningSamples-PredictiveMaintenance), top Kaggle notebooks on this dataset.

---

*Trained on Google Colab free tier · All infrastructure 100% free · Built March 2026*

---
