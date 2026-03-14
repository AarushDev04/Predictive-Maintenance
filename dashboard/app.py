"""Streamlit Predictive Maintenance Dashboard
Run locally:  streamlit run dashboard/app.py
Deploy:       streamlit.io/cloud → New app → connect GitHub repo
"""
import json, pickle, os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='PredMaint Dashboard',
    page_icon='🔧',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp                { background-color: #0c0b08; color: #d0c8b8; }
    .block-container      { padding-top: 2rem; }
    div[data-testid="metric-container"] {
        background: #111008;
        border: 1px solid #2a2510;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .risk-critical { color: #ff3d3d; font-size: 1.6rem; font-weight: 700; }
    .risk-high     { color: #ff6b35; font-size: 1.6rem; font-weight: 700; }
    .risk-medium   { color: #f0a500; font-size: 1.6rem; font-weight: 700; }
    .risk-low      { color: #7dff6b; font-size: 1.6rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SENSORS       = ['volt', 'rotate', 'pressure', 'vibration']
SENSOR_COLORS = {
    'volt':      '#f0a500',
    'rotate':    '#00d4ff',
    'pressure':  '#7dff6b',
    'vibration': '#ff6b35',
}
PLOTLY_THEME = dict(
    paper_bgcolor='#0c0b08',
    plot_bgcolor='#111008',
    font=dict(color='#a09888'),
)

DATA_PATH     = os.getenv('DATA_PATH',     'data/raw')
REGISTRY_PATH = 'models/model_registry.json'
STATUS_PATH   = 'models/status.json'
FEATURES_PATH = 'models/features.json'
MODEL_PATH    = 'models/lgbm_v1.pkl'
THRESHOLD_PATH= 'models/threshold.json'

# ── Feature engineering (must mirror training) ────────────────────────────────
def engineer_features_serving(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('datetime').copy()
    WINDOWS, LAGS = [3, 6, 12, 24], [1, 3, 6]
    for s in SENSORS:
        for w in WINDOWS:
            df[f'{s}_roll{w}_mean'] = df[s].rolling(w, min_periods=1).mean()
            df[f'{s}_roll{w}_std']  = df[s].rolling(w, min_periods=1).std().fillna(0)
        for lag in LAGS:
            df[f'{s}_lag{lag}h'] = df[s].shift(lag).fillna(df[s].median())
        df[f'{s}_fft_dominant']    = 0.0
        df[f'{s}_prophet_anomaly'] = 0.0
    return df

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner='Loading sensor data...')
def load_data():
    tel  = pd.read_csv(f'{DATA_PATH}/PdM_telemetry.csv', parse_dates=['datetime'])
    fail = pd.read_csv(f'{DATA_PATH}/PdM_failures.csv',  parse_dates=['datetime'])
    return tel, fail

@st.cache_resource(show_spinner='Loading model...')
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURES_PATH) as f:
        feature_cols = json.load(f)
    with open(THRESHOLD_PATH) as f:
        threshold = json.load(f)['threshold']
    return model, feature_cols, threshold

def load_json_safe(path: str, default: dict) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default

# ── Load everything ───────────────────────────────────────────────────────────
try:
    telemetry, failures = load_data()
    model, feature_cols, threshold = load_model()
    DATA_LOADED = True
except Exception as e:
    DATA_LOADED = False
    st.error(f'⚠️  Could not load data or model: {e}')
    st.info('Run the training notebook first to generate model files.')
    st.stop()

registry = load_json_safe(REGISTRY_PATH, {})
status   = load_json_safe(STATUS_PATH,   {})

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## 🔧 PredMaint')
    st.caption('Predictive Maintenance Dashboard')
    st.divider()

    machine_ids      = sorted(telemetry['machineID'].unique())
    selected_machine = st.selectbox(
        'Select Machine',
        machine_ids,
        format_func=lambda x: f'Machine {x:03d}',
    )

    st.divider()
    st.caption('MODEL INFO')
    prod = registry.get('production', {})
    st.markdown(f"**Version:** `{prod.get('version', 'N/A')}`")
    st.markdown(f"**AUC:** `{prod.get('mean_auc', 0):.4f}`")
    trained = str(prod.get('trained_at', 'N/A'))[:10]
    st.markdown(f'**Trained:** `{trained}`')

    st.divider()
    st.caption('DRIFT STATUS')
    if status.get('drift_detected'):
        st.error('⚠️ Data drift detected')
        st.caption(f"Drifted: {status.get('n_drifted_columns','?')}"
                   f"/{status.get('n_total_columns','?')} columns")
    else:
        st.success('✅ No drift detected')
    st.caption(f"Last check: {status.get('last_drift_check', 'Never')}")
    if status.get('retrain_needed'):
        st.warning('🔄 Retraining recommended')

    st.divider()
    if st.button('🔄 Clear cache & reload'):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# ── Machine data ──────────────────────────────────────────────────────────────
m_tel   = (telemetry[telemetry['machineID'] == selected_machine]
           .sort_values('datetime').reset_index(drop=True))
m_fails = failures[failures['machineID'] == selected_machine]

# ── Compute current failure probability ───────────────────────────────────────
recent = m_tel.tail(24).copy()
try:
    feat_df      = engineer_features_serving(recent)
    # Align to training feature set
    for col in feature_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    X_now        = feat_df[feature_cols].tail(1)
    current_prob = float(model.predict_proba(X_now)[0, 1])
except Exception as e:
    st.warning(f'Prediction error: {e}')
    current_prob = 0.0

risk  = ('CRITICAL' if current_prob >= 0.75 else
         'HIGH'     if current_prob >= 0.50 else
         'MEDIUM'   if current_prob >= 0.25 else 'LOW')
gauge_color = {'CRITICAL': '#ff3d3d', 'HIGH': '#ff6b35',
               'MEDIUM': '#f0a500', 'LOW': '#7dff6b'}[risk]

# ═════════════════════════════════════════════════════════════════════════════
# ROW 1 — Title + Quick metrics + Gauge
# ═════════════════════════════════════════════════════════════════════════════
col_title, col_gauge = st.columns([2.2, 1])

with col_title:
    st.markdown(f'## Machine {selected_machine:03d} — Failure Risk Dashboard')
    st.caption(f'Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    st.markdown('')

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric('Total Failures',  len(m_fails))
    mc2.metric('Sensor Readings', f'{len(m_tel):,}')
    mc3.metric('Data Since',      str(m_tel['datetime'].min().date()))
    mc4.metric('Last Reading',    str(m_tel['datetime'].max().date()))

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode='gauge+number',
        value=round(current_prob * 100, 1),
        number={'suffix': ' %', 'font': {'color': '#f0e8d0', 'size': 36}},
        title={
            'text': f'48h Failure Risk<br><b class="risk-{risk.lower()}">{risk}</b>',
            'font': {'color': '#d0c8b8', 'size': 13},
        },
        gauge={
            'axis':        {'range': [0, 100], 'tickcolor': '#60584a', 'tickfont': {'color': '#60584a'}},
            'bar':         {'color': gauge_color, 'thickness': 0.25},
            'bgcolor':     '#111008',
            'bordercolor': '#2a2510',
            'steps': [
                {'range': [0,  25], 'color': '#0a1a08'},
                {'range': [25, 50], 'color': '#1a1a08'},
                {'range': [50, 75], 'color': '#1a1008'},
                {'range': [75,100], 'color': '#1a0808'},
            ],
            'threshold': {
                'line':  {'color': '#ff3d3d', 'width': 2},
                'value': threshold * 100,
            },
        },
    ))
    fig_gauge.update_layout(
        height=230, margin=dict(t=50, b=10, l=20, r=20),
        **PLOTLY_THEME,
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# ROW 2 — Sensor trends (last 7 days)
# ═════════════════════════════════════════════════════════════════════════════
st.subheader('📡 Sensor Trends — Last 7 Days')

last7d       = m_tel[m_tel['datetime'] >= m_tel['datetime'].max() - timedelta(days=7)]
window_fails = m_fails[m_fails['datetime'] >= m_tel['datetime'].max() - timedelta(days=7)]

fig_sensors = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    subplot_titles=[s.capitalize() for s in SENSORS],
    vertical_spacing=0.06,
)
for i, (sensor, color) in enumerate(SENSOR_COLORS.items(), start=1):
    fig_sensors.add_trace(
        go.Scatter(
            x=last7d['datetime'], y=last7d[sensor],
            line=dict(color=color, width=1.2),
            name=sensor.capitalize(),
            hovertemplate=f'%{{x}}<br>{sensor}: %{{y:.2f}}<extra></extra>',
        ),
        row=i, col=1,
    )
    for _, fr in window_fails.iterrows():
        fig_sensors.add_vline(
            x=fr['datetime'], line_color='#ff3d3d',
            line_dash='dash', line_width=1.2,
            row=i, col=1,
        )

fig_sensors.update_layout(
    height=520, showlegend=True,
    legend=dict(orientation='h', y=1.02, font=dict(color='#a09888')),
    margin=dict(t=40, b=20),
    **PLOTLY_THEME,
)
fig_sensors.update_xaxes(gridcolor='#1a1910', showgrid=True)
fig_sensors.update_yaxes(gridcolor='#1a1910', showgrid=True)
st.plotly_chart(fig_sensors, use_container_width=True)

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# ROW 3 — Failure history + Recent readings table
# ═════════════════════════════════════════════════════════════════════════════
col_hist, col_table = st.columns([2, 1])

with col_hist:
    st.subheader('📋 Failure History')
    if len(m_fails) > 0:
        fig_hist = px.scatter(
            m_fails, x='datetime', y='failure',
            color='failure',
            color_discrete_map={
                'comp1': '#f0a500', 'comp2': '#00d4ff',
                'comp3': '#7dff6b', 'comp4': '#ff6b35',
            },
            title=f'Failure Timeline — Machine {selected_machine:03d}',
            template='plotly_dark',
        )
        fig_hist.update_traces(marker=dict(size=10, symbol='x'))
        fig_hist.update_layout(
            height=260, margin=dict(t=40, b=20),
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info('No historical failures recorded for this machine.')

with col_table:
    st.subheader('Recent Readings')
    display_df = (
        m_tel[['datetime'] + SENSORS]
        .tail(10)
        .sort_values('datetime', ascending=False)
        .reset_index(drop=True)
    )
    display_df['datetime'] = display_df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(display_df, use_container_width=True, height=290)

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# ROW 4 — SHAP attribution
# ═════════════════════════════════════════════════════════════════════════════
st.subheader('🔍 SHAP Feature Attribution — Why This Risk Score?')
st.caption('Which sensor readings pushed the failure probability up or down.')

col_shap, col_action = st.columns([2, 1])

with col_shap:
    try:
        import shap as shap_lib
        explainer = shap_lib.TreeExplainer(model)
        shap_vals = explainer(X_now).values[0]

        TOP_N    = 12
        abs_vals = np.abs(shap_vals)
        top_idx  = np.argsort(abs_vals)[-TOP_N:][::-1]
        feats    = [feature_cols[i] for i in top_idx]
        vals     = [float(shap_vals[i]) for i in top_idx]
        colors   = ['#ff6b35' if v > 0 else '#7dff6b' for v in vals]

        fig_shap = go.Figure(go.Bar(
            x=vals[::-1],
            y=feats[::-1],
            orientation='h',
            marker_color=colors[::-1],
            text=[f'{v:+.3f}' for v in vals[::-1]],
            textposition='outside',
            hovertemplate='%{y}<br>SHAP: %{x:.4f}<extra></extra>',
        ))
        fig_shap.update_layout(
            title='SHAP Values  ·  🟠 increases risk   🟢 decreases risk',
            height=400,
            xaxis=dict(title='SHAP value', gridcolor='#1a1910', zeroline=True,
                       zerolinecolor='#3a3020', zerolinewidth=1),
            margin=dict(t=50, b=20, l=20, r=60),
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    except Exception as e:
        st.warning(f'SHAP computation unavailable: {e}')

with col_action:
    st.markdown('**How to interpret**')
    st.markdown('- 🟠 **Orange** → sensor pushing risk **up**')
    st.markdown('- 🟢 **Green** → sensor pushing risk **down**')
    st.markdown('- Longer bar = stronger influence')
    st.divider()

    st.markdown('**Recommended action**')
    if risk == 'CRITICAL':
        st.error(f'🚨 Immediate inspection required
Machine {selected_machine:03d}')
    elif risk == 'HIGH':
        st.error(f'⚠️ Schedule inspection today
Machine {selected_machine:03d}')
    elif risk == 'MEDIUM':
        st.warning(f'📅 Schedule inspection within 48h
Machine {selected_machine:03d}')
    else:
        st.success('✅ No action required')

    st.divider()
    st.markdown('**Current threshold**')
    st.caption(f'Alert fires at probability ≥ `{threshold:.3f}`')
    st.caption(f'Current probability: `{current_prob:.4f}`')
    st.caption(f'Margin: `{abs(current_prob - threshold):+.4f}`')