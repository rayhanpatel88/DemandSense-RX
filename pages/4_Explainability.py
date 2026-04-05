"""Page 5 — Explainability (SHAP)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config import load_config
from src.pipeline import run_pipeline

st.set_page_config(page_title="Explainability · DemandSense-RX",
                   page_icon="🔍", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f1f35; }
[data-testid="stSidebar"] * { color: #cdd9e5 !important; }
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading pipeline…")
def get_data():
    return run_pipeline(load_config())


data = get_data()
shap_data = data["shap_data"]
feature_importance = data["feature_importance"]
train_df = data["train_df"]
feature_cols = data["feature_cols"]

with st.sidebar:
    st.title("🔍 Explainability")
    st.divider()
    top_n = st.slider("Top N Features", 5, 30, 15)
    st.divider()
    st.page_link("streamlit_app.py", label="← Executive Overview")

st.title("🔍 Model Explainability")
st.caption("Global feature importance · SHAP value distribution · Single prediction drilldown")
st.divider()

# ── Global feature importance (LightGBM built-in) ─────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 LightGBM Feature Importance")
    st.caption("Based on gain (contribution to reducing loss)")
    top_fi = feature_importance.head(top_n)
    fig = px.bar(top_fi, x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale="Blues",
                 template="plotly_dark", height=420)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,45,0.8)",
        margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False, xaxis_title="Importance Score", yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    if shap_data and "global_importance" in shap_data:
        st.subheader("🎯 SHAP Global Feature Importance")
        st.caption("Mean |SHAP value| — average impact on model output magnitude")
        shap_global = shap_data["global_importance"].head(top_n)
        fig2 = px.bar(shap_global, x="importance", y="feature", orientation="h",
                      color="importance", color_continuous_scale="Reds",
                      template="plotly_dark", height=420)
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,45,0.8)",
            margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False, xaxis_title="Mean |SHAP|", yaxis_title="",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.subheader("🎯 SHAP Feature Importance")
        st.info("SHAP values not available. Check that the model trained successfully.")

st.divider()

# ── SHAP Distribution (beeswarm-style) ───────────────────────────────────────
if shap_data and "shap_df" in shap_data and "sample_df" in shap_data:
    st.subheader("📈 SHAP Value Distribution (Top Features)")
    st.caption("Each point = one prediction sample. Positive values push forecast higher.")

    shap_df = shap_data["shap_df"]
    sample_df = shap_data["sample_df"]
    top_features = shap_data["global_importance"].head(8)["feature"].tolist()

    rows = []
    for feat in top_features:
        if feat in shap_df.columns and feat in sample_df.columns:
            for i in range(len(shap_df)):
                rows.append({
                    "feature": feat,
                    "shap_value": shap_df[feat].iloc[i],
                    "feature_value": sample_df[feat].iloc[i],
                })
    if rows:
        beeswarm_df = pd.DataFrame(rows)
        fig3 = px.strip(beeswarm_df, x="shap_value", y="feature",
                        color="feature_value", color_continuous_scale="RdBu_r",
                        template="plotly_dark", height=350,
                        labels={"shap_value": "SHAP Value", "feature": "Feature"})
        fig3.add_vline(x=0, line_dash="dash", line_color="white")
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,45,0.8)",
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(title="Feature<br>Value"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── Single prediction explanation ─────────────────────────────────────────
    st.subheader("🔎 Single Prediction Explanation")
    n_samples = len(shap_df)
    sample_idx = st.slider("Select Sample Index", 0, max(n_samples - 1, 1), 0)

    explainer = shap_data.get("explainer")
    if explainer:
        local = explainer.local_explanation(sample_idx)
        top_local = local.head(12)

        col_l, col_r = st.columns([2, 1])
        with col_l:
            colors = ["#ff4444" if v > 0 else "#4444ff" for v in top_local["shap_value"]]
            fig4 = go.Figure(go.Bar(
                x=top_local["shap_value"],
                y=top_local["feature"],
                orientation="h",
                marker_color=colors,
            ))
            fig4.add_vline(x=0, line_dash="solid", line_color="white")
            fig4.update_layout(
                title="SHAP Waterfall (Top 12 Features)",
                template="plotly_dark", height=380,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,45,0.8)",
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis=dict(autorange="reversed"),
                xaxis_title="SHAP Value", yaxis_title="",
            )
            st.plotly_chart(fig4, use_container_width=True)

        with col_r:
            st.markdown("**Feature Values for This Sample**")
            sample_vals = local[["feature", "feature_value", "shap_value"]].head(12).copy()
            sample_vals["shap_value"] = sample_vals["shap_value"].round(3)
            sample_vals["feature_value"] = sample_vals["feature_value"].round(3)
            st.dataframe(sample_vals, use_container_width=True, hide_index=True)
else:
    st.info("SHAP explainability data not available. The model may need to retrain with SHAP support.")
