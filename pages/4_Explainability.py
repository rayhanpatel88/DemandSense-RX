"""Explainability page."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from src.app.ui import apply_page_config, get_explainability_data, get_pipeline_data, render_header, render_sidebar, style_plotly
except ImportError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.app.ui import apply_page_config, get_explainability_data, get_pipeline_data, render_header, render_sidebar, style_plotly

apply_page_config("Explainability")
base_data = get_pipeline_data()
render_sidebar("explainability", base_data)
data = get_explainability_data()

shap_data = data.get("shap_data", {})
feature_importance = data["feature_importance"]

with st.sidebar:
    st.divider()
    top_n = st.slider("Top features", 6, 20, 12)
    sample_idx = st.slider("Example row", 0, max(len(shap_data.get("sample_df", [])) - 1, 0), 0)

render_header(
    "Model Introspection",
    "Explainability",
    "This page explains which inputs matter most to the forecast so you can see what the model is paying attention to.",
)

left, right = st.columns(2, gap="large")
with left:
    fig = px.bar(feature_importance.head(top_n), x="importance", y="feature", orientation="h", color="importance", color_continuous_scale=["#dfe8f1", "#1f3a5f"])
    fig = style_plotly(fig, 420)
    fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"}, yaxis_title="", xaxis_title="How strongly this input influences the forecast")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Which Inputs Matter Most")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("What This Means In Business Terms")
    st.markdown(
        "<div class='note'>High-ranking lag and rolling features confirm the model is learning persistence structure. "
        "Price-gap and promotion terms show explicit commercial sensitivity instead of pretending demand is purely seasonal.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    if shap_data and "global_importance" in shap_data:
        shap_fig = px.bar(shap_data["global_importance"].head(top_n), x="importance", y="feature", orientation="h", color="importance", color_continuous_scale=["#ead9cf", "#a64032"])
        shap_fig = style_plotly(shap_fig, 320)
        shap_fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"}, yaxis_title="", xaxis_title="Average strength of impact on the forecast")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Inputs With The Biggest Overall Impact")
        st.plotly_chart(shap_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

if shap_data and "shap_df" in shap_data and "sample_df" in shap_data:
    shap_df = shap_data["shap_df"]
    sample_df = shap_data["sample_df"]
    top_features = shap_data["global_importance"].head(min(8, top_n))["feature"].tolist()
    rows = []
    for feature in top_features:
        for idx in range(len(shap_df)):
            rows.append({"feature": feature, "shap_value": shap_df.iloc[idx][feature], "feature_value": sample_df.iloc[idx][feature]})
    beeswarm = px.scatter(rows, x="shap_value", y="feature", color="feature_value", color_continuous_scale=["#1f3a5f", "#d9c9ae", "#a64032"])
    beeswarm = style_plotly(beeswarm, 360)
    beeswarm.add_vline(x=0, line_dash="dash", line_color="#6d685f")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("How Each Input Pushes Forecasts Up Or Down")
    st.plotly_chart(beeswarm, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    explainer = shap_data.get("explainer")
    if explainer:
        local = explainer.local_explanation(sample_idx).head(12)
        waterfall = go.Figure(
            go.Bar(
                x=local["shap_value"],
                y=local["feature"],
                orientation="h",
                marker_color=["#a64032" if value > 0 else "#1f3a5f" for value in local["shap_value"]],
            )
        )
        waterfall = style_plotly(waterfall, 360)
        waterfall.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="", xaxis_title="How much this input moves the forecast")
        info, panel = st.columns([1.3, 1.0], gap="large")
        with info:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Why One Specific Forecast Looks The Way It Does")
            st.plotly_chart(waterfall, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with panel:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Input Values Behind That Forecast")
            local_display = local[["feature", "feature_value", "shap_value"]].rename(
                columns={
                    "feature": "Input",
                    "feature_value": "Value Used",
                    "shap_value": "Impact On Forecast",
                }
            )
            st.dataframe(local_display, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
