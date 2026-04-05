"""Backtesting page."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from src.app.ui import apply_page_config, get_pipeline_data, render_header, render_sidebar, style_plotly
except ImportError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.app.ui import apply_page_config, get_pipeline_data, render_header, render_sidebar, style_plotly

apply_page_config("Backtesting")
data = get_pipeline_data()
render_sidebar("backtesting", data)

summary = data["backtest_results"]["summary"]
by_sku = data["backtest_results"]["by_sku"]
predictions = data["backtest_results"]["predictions"]

with st.sidebar:
    st.divider()
    metric = st.selectbox("Primary metric", ["WAPE", "RMSE", "MAE", "MAPE"], index=0)

render_header(
    "Model Evaluation",
    "Backtesting",
    "Rolling-origin backtests now mirror recursive inference, so the comparison reflects the system that actually runs instead of a one-step approximation.",
)

if summary.empty:
    st.warning("Backtesting results were unavailable.")
    st.stop()

left, right = st.columns([1.0, 1.4], gap="large")
with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Model Summary")
    st.dataframe(summary.reset_index(), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    comp = px.bar(summary.reset_index().melt(id_vars="model", var_name="metric", value_name="value"), x="metric", y="value", color="model", barmode="group", color_discrete_map={"LightGBM": "#1f3a5f", "MovingAverage": "#8c5a2b", "SeasonalNaive": "#6d685f"})
    comp = style_plotly(comp, 320)
    comp.update_layout(legend=dict(orientation="h", y=1.08, x=0), xaxis_title="", yaxis_title="Metric value")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Cross-Model Comparison")
    st.plotly_chart(comp, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if not predictions.empty:
    preds = predictions.copy()
    preds["abs_error"] = (preds["demand"] - preds["forecast"]).abs()
    error_time = preds.groupby(["date", "model"])["abs_error"].mean().reset_index()
    line = px.line(error_time, x="date", y="abs_error", color="model", color_discrete_map={"LightGBM": "#1f3a5f", "MovingAverage": "#8c5a2b", "SeasonalNaive": "#6d685f"})
    line = style_plotly(line, 320)
    line.update_layout(legend=dict(orientation="h", y=1.08, x=0), yaxis_title="Mean absolute error", xaxis_title="")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Error Through Time")
    st.plotly_chart(line, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    lgbm = preds[preds["model"] == "LightGBM"].copy()
    sample = lgbm.sample(min(500, len(lgbm)), random_state=42) if not lgbm.empty else lgbm
    if not sample.empty:
        scatter = px.scatter(sample, x="demand", y="forecast", opacity=0.55, color_discrete_sequence=["#1f3a5f"])
        max_val = max(sample["demand"].max(), sample["forecast"].max())
        scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="#a64032", dash="dash"))
        scatter = style_plotly(scatter, 330)
        scatter.update_layout(showlegend=False, xaxis_title="Actual demand", yaxis_title="Predicted demand")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("LightGBM Calibration")
        st.plotly_chart(scatter, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

if not by_sku.empty:
    focus = by_sku.pivot(index="sku", columns="model", values=metric).reset_index()
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader(f"Per-SKU {metric}")
    st.dataframe(focus.sort_values(by=focus.columns[1], ascending=False), use_container_width=True, hide_index=True, height=360)
    st.markdown("</div>", unsafe_allow_html=True)
