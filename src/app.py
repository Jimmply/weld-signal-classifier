"""Weld In-Process Signal Classifier Dashboard — Streamlit app."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from signal_generator import (
    DEFECT_TYPES, SIGNAL_COLS, WeldFleetSignalGenerator, WeldSignalGenerator,
)
from classifier import WeldSignalClassifier

st.set_page_config(page_title="Weld Signal Monitor", page_icon="📡", layout="wide")

DEFECT_COLORS = {
    "Good":           "#2ecc71",
    "Spatter":        "#e74c3c",
    "Porosity":       "#9b59b6",
    "Cracking":       "#e67e22",
    "Lack_of_Fusion": "#3498db",
}


@st.cache_resource
def load_model():
    df = WeldFleetSignalGenerator(n_welds=600).generate_summary()
    clf = WeldSignalClassifier()
    results = clf.fit(df)
    return df, clf, results


df, clf, results = load_model()

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.title("📡 Weld Signal Monitor")
st.sidebar.markdown("Real-time photodiode + acoustic emission analysis for Nd:YAG laser welding.")

selected_defect = st.sidebar.selectbox(
    "Simulate defect type", DEFECT_TYPES, index=0
)
seed = st.sidebar.slider("Signal seed (variation)", 0, 99, 42)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Classifier accuracy:** {results.test_accuracy:.1%}")
st.sidebar.markdown(f"**Training welds:** {len(df):,}")

# ── Simulate a weld ────────────────────────────────────────────
gen = WeldSignalGenerator(defect_type=selected_defect, random_seed=seed)
ws  = gen.generate(weld_id="LIVE-WELD")
sig = ws.df

# Classify from extracted features
feats = WeldFleetSignalGenerator()._extract_features(ws)
feat_row = pd.DataFrame([feats])
predicted, confidence, shap_df = clf.predict_with_explanation(feat_row)

# ── Header ─────────────────────────────────────────────────────
st.title("Laser Weld In-Process Signal Classifier")
st.markdown(
    "Simulates 10 kHz photodiode, acoustic emission, and back-reflection signals "
    "captured during a 500 ms Nd:YAG weld pass. XGBoost classifies defect type in real time."
)

color = DEFECT_COLORS.get(predicted, "#999")
c1, c2, c3 = st.columns(3)
c1.markdown(
    f"<div style='background:{color};padding:16px;border-radius:8px;text-align:center;"
    f"color:white;font-weight:bold;font-size:20px'>Predicted: {predicted}</div>",
    unsafe_allow_html=True,
)
c2.metric("Confidence", f"{confidence:.1%}")
c3.metric("Event Rate", f"{sig['is_event'].mean():.1%}")

st.markdown("---")

# ── Signal traces ──────────────────────────────────────────────
st.subheader("Live Signal Traces (500 ms weld pass)")

# Downsample for display (10 kHz → 1 kHz)
step = 10
sig_ds = sig.iloc[::step].copy()
event_rows = sig_ds[sig_ds["is_event"]]

signal_labels = {
    "photodiode_v":       "Photodiode (V) — melt pool emission",
    "acoustic_rms_mv":    "Acoustic RMS (mV) — spatter / cracking",
    "back_reflection_pct":"Back-Reflection (%) — keyhole stability",
}

for col, label in signal_labels.items():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sig_ds["time_s"], y=sig_ds[col],
        mode="lines", line=dict(width=1, color="#2c3e50"),
        name=label,
    ))
    if len(event_rows) > 0:
        fig.add_trace(go.Scatter(
            x=event_rows["time_s"], y=event_rows[col],
            mode="markers",
            marker=dict(color="#e74c3c", size=3, symbol="x"),
            name="Anomaly event",
        ))
    fig.update_layout(
        title=label, height=180, showlegend=False,
        margin=dict(l=30, r=10, t=35, b=25),
        xaxis_title="Time (s)", yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Feature importance + class probabilities ───────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Defect Class Probabilities")
    proba_df = clf.class_probabilities(feat_row)
    fig_p = px.bar(
        proba_df, x="probability", y="defect_type", orientation="h",
        color="defect_type", color_discrete_map=DEFECT_COLORS,
        range_x=[0, 1],
    )
    fig_p.update_layout(showlegend=False, height=280, margin=dict(t=10))
    st.plotly_chart(fig_p, use_container_width=True)

with col2:
    st.subheader("Feature Importance")
    fi = results.feature_importances.tail(10)
    fig_fi = px.bar(fi, orientation="h", color=fi.values,
                    color_continuous_scale="Reds",
                    labels={"value": "Importance", "index": ""})
    fig_fi.update_layout(showlegend=False, coloraxis_showscale=False,
                         height=280, margin=dict(t=10))
    st.plotly_chart(fig_fi, use_container_width=True)

# ── Fleet defect distribution ──────────────────────────────────
st.markdown("---")
st.subheader("Training Fleet — Defect Distribution")
dist = df["defect_type"].value_counts().reset_index()
dist.columns = ["Defect Type", "Count"]
fig_d = px.bar(dist, x="Defect Type", y="Count",
               color="Defect Type", color_discrete_map=DEFECT_COLORS)
fig_d.update_layout(showlegend=False, height=280)
st.plotly_chart(fig_d, use_container_width=True)
