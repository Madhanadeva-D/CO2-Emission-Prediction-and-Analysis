import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CO₂ Emission Predictor",
    page_icon="🌿",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.block-container {
    padding-top: 0.7rem !important;
    padding-bottom: 2rem !important;
}

.stApp {
    background: linear-gradient(135deg, #f0faf2 0%, #e8f5eb 50%, #f4fdf6 100%);
    color: #1a3a1e;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e6f4ea 0%, #d4edda 100%);
    border-right: 1px solid #a5d6a7;
}
[data-testid="stSidebar"] * { color: #2e7d32 !important; }

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #a5d6a7;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 12px rgba(46,125,50,0.10);
}
/* Metric cards — label 11px (standard caption), value 24px (standard headline) */
[data-testid="stMetricLabel"] { color: #388e3c !important; font-size: 0.6875rem; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #1b5e20 !important; font-family: 'DM Mono', monospace; font-size: 1.5rem; }

/* Slider label — 14px body standard */
[data-testid="stSlider"] > div > div > div > div { background: #2e7d32 !important; }
[data-testid="stSlider"] label { color: #2e7d32 !important; font-size: 0.875rem; letter-spacing: 0.01em; }

.prediction-box {
    background: linear-gradient(135deg, #2e7d32 0%, #388e3c 100%);
    border: 1px solid #66bb6a;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(46,125,50,0.18);
    margin: 16px 0;
}
/* Prediction label — 12px overline/caption standard */
.prediction-box .label { font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; color: #c8e6c9; margin-bottom: 8px; }
/* Prediction value — 48px display number, clearly primary */
.prediction-box .value { font-family: 'DM Mono', monospace; font-size: 3rem; font-weight: 700; color: #ffffff; line-height: 1; }
/* Unit text — 14px supporting body */
.prediction-box .unit  { font-size: 0.875rem; color: #a5d6a7; margin-top: 6px; }

/* Badge — 13px chip/tag standard */
.badge { display: inline-block; border-radius: 30px; padding: 5px 16px; font-size: 0.8125rem; font-weight: 600; letter-spacing: 0.04em; margin-top: 14px; }
.badge-low    { background: #e8f5e9; color: #1b5e20; border: 1px solid #4caf50; }
.badge-medium { background: #fff3e0; color: #e65100; border: 1px solid #ff9800; }
.badge-high   { background: #ffebee; color: #b71c1c; border: 1px solid #f44336; }

/* Section title — 11px overline standard */
.section-title { font-size: 0.6875rem; letter-spacing: 0.14em; text-transform: uppercase; color: #2e7d32; margin-bottom: 12px; padding-bottom: 6px; border-bottom: 2px solid #a5d6a7; }

/* Info rows — 14px body standard, mono value for data legibility */
.info-row { display: flex; justify-content: space-between; padding: 9px 10px; border-bottom: 1px solid #dcedc8; font-size: 0.875rem; background: #fff; border-radius: 4px; margin-bottom: 2px; }
.info-row .key { color: #388e3c; font-weight: 400; }
.info-row .val { font-family: 'DM Mono', monospace; color: #1b5e20; font-size: 0.875rem; font-weight: 500; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load & train ──────────────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv('co2 Emissions.csv')

    fuel_map = {"Z": "Premium Gasoline", "X": "Regular Gasoline",
                "D": "Diesel", "E": "Ethanol(E85)", "N": "Natural Gas"}
    df["Fuel Type"] = df["Fuel Type"].map(fuel_map)
    df = df[~df["Fuel Type"].str.contains("Natural Gas", na=False)].reset_index(drop=True)

    df_model = df[['Engine Size(L)', 'Cylinders',
                   'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']].copy()
    df_model = df_model[(np.abs(stats.zscore(df_model)) < 1.9).all(axis=1)]

    X = df_model[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']]
    y = df_model['CO2 Emissions(g/km)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2     = r2_score(y_test, y_pred)
    mae    = mean_absolute_error(y_test, y_pred)

    return model, r2, mae, float(y.min()), float(y.max()), float(y.mean())

model, r2, mae, y_min, y_max, y_mean = load_and_train()


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# CO₂ Emission Predictor")
st.markdown(f'<p style="color:#388e3c;font-size:0.875rem;margin-top:-12px;margin-bottom:24px;letter-spacing:0.01em;">Gradient Boosting · R² = {r2:.4f}</p>', unsafe_allow_html=True)

col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.markdown('<div class="section-title">Vehicle Parameters</div>', unsafe_allow_html=True)

    engine_size = st.slider('Engine Size (L)',                      min_value=0.9,  max_value=8.4,  value=3.0,  step=0.1)
    cylinders   = st.slider('Cylinders',                            min_value=3,    max_value=16,   value=6,    step=1)
    fuel_comb   = st.slider('Fuel Consumption Comb (L/100 km)',     min_value=4.0,  max_value=26.0, value=10.5, step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Min CO₂", f"{y_min:.0f}", "g/km")
    m2.metric("Avg CO₂", f"{y_mean:.0f}", "g/km")
    m3.metric("Max CO₂", f"{y_max:.0f}", "g/km")

with col_right:
    input_df = pd.DataFrame([[engine_size, cylinders, fuel_comb]],
                             columns=['Engine Size(L)', 'Cylinders',
                                      'Fuel Consumption Comb (L/100 km)'])
    prediction = float(model.predict(input_df)[0])
    prediction = max(y_min, min(prediction, y_max))

    if prediction < 150:
        level, badge_class, emoji = "Low Emission",    "badge-low",    "🟢"
    elif prediction <= 250:
        level, badge_class, emoji = "Medium Emission", "badge-medium", "🟡"
    else:
        level, badge_class, emoji = "High Emission",   "badge-high",   "🔴"

    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="prediction-box">
        <div class="label">Estimated CO₂ Output</div>
        <div class="value">{prediction:.1f}</div>
        <div class="unit">grams per kilometre</div>
        <div><span class="badge {badge_class}">{emoji} {level}</span></div>
    </div>
    """, unsafe_allow_html=True)

    gauge = (prediction - y_min) / (y_max - y_min)
    st.markdown('<span style="font-size:0.6875rem;color:#2e7d32;letter-spacing:0.12em;text-transform:uppercase;">Emission Gauge</span>', unsafe_allow_html=True)
    st.progress(min(gauge, 1.0))
    st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#81c784;font-family:monospace;margin-top:4px;"><span>{y_min:.0f}</span><span>{y_max:.0f} g/km</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Input Summary</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-row"><span class="key">Engine Size</span><span class="val">{engine_size} L</span></div>
    <div class="info-row"><span class="key">Cylinders</span><span class="val">{cylinders}</span></div>
    <div class="info-row"><span class="key">Fuel Consumption</span><span class="val">{fuel_comb} L/100km</span></div>
    <div class="info-row"><span class="key">Predicted CO₂</span><span class="val">{prediction:.2f} g/km</span></div>
    """, unsafe_allow_html=True)

st.markdown("""
<hr style="border:1px solid #a5d6a7; margin-top:10px;">

<div style="text-align:center; font-size:0.875rem; color:#388e3c;">
    Developed by <b>Madhanadeva D</b> · 
    <a href="https://github.com/Madhanadeva-D/Madhanadeva-D" target="_blank" style="color:#2e7d32; text-decoration:none;">
        GitHub
    </a>
</div>
""", unsafe_allow_html=True)