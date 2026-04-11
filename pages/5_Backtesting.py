"""Backtesting page."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from src.app.ui import apply_page_config, get_backtesting_data, get_pipeline_data, render_header, render_sidebar, style_plotly
except ImportError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.app.ui import apply_page_config, get_backtesting_data, get_pipeline_data, render_header, render_sidebar, style_plotly

apply_page_config("Backtesting")
base_data = get_pipeline_data()
render_sidebar("backtesting", base_data)
data = get_backtesting_data()

summary = data["backtest_results"]["summary"]
by_sku = data["backtest_results"]["by_sku"]
predictions = data["backtest_results"]["predictions"]
metric_labels = {
    "WAPE": "Average % Forecast Miss",
    "RMSE": "Typical Forecast Error",
    "MAE": "Average Units Missed",
    "MAPE": "Average % Error",
}
model_labels = {
    "LightGBM": "Main AI Model",
    "MovingAverage": "Moving Average",
    "SeasonalNaive": "Seasonal Baseline",
}

with st.sidebar:
    st.divider()
    metric = st.selectbox("Accuracy measure", ["WAPE", "RMSE", "MAE", "MAPE"], index=0, format_func=lambda x: metric_labels[x])

render_header(
    "Model Evaluation",
    "Backtesting",
    "This page compares forecast models against past actuals so you can see which approach has been the most accurate.",
)

if summary.empty:
    st.warning("Backtesting results were unavailable.")
    st.stop()

left, right = st.columns([1.0, 1.4], gap="large")
with left:
    summary_display = summary.reset_index().rename(columns={"model": "Forecast Model"}).replace({"Forecast Model": model_labels})
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Accuracy Summary")
    st.dataframe(summary_display, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    comparison_df = summary.reset_index().melt(id_vars="model", var_name="metric", value_name="value")
    comparison_df["model"] = comparison_df["model"].map(model_labels)
    comparison_df["metric"] = comparison_df["metric"].map(metric_labels).fillna(comparison_df["metric"])
    comp = px.bar(comparison_df, x="metric", y="value", color="model", barmode="group", color_discrete_map={"Main AI Model": "#1f3a5f", "Moving Average": "#8c5a2b", "Seasonal Baseline": "#6d685f"})
    comp = style_plotly(comp, 320)
    comp.update_layout(legend=dict(orientation="h", y=1.08, x=0), xaxis_title="", yaxis_title="Accuracy score")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Which Forecast Model Performs Best")
    st.plotly_chart(comp, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if not predictions.empty:
    preds = predictions.copy()
    preds["abs_error"] = (preds["demand"] - preds["forecast"]).abs()
    preds["model"] = preds["model"].map(model_labels).fillna(preds["model"])
    error_time = preds.groupby(["date", "model"])["abs_error"].mean().reset_index()
    line = px.line(error_time, x="date", y="abs_error", color="model", color_discrete_map={"Main AI Model": "#1f3a5f", "Moving Average": "#8c5a2b", "Seasonal Baseline": "#6d685f"})
    line = style_plotly(line, 320)
    line.update_layout(legend=dict(orientation="h", y=1.08, x=0), yaxis_title="Average forecast miss", xaxis_title="")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("How Accuracy Changed Over Time")
    st.plotly_chart(line, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    lgbm = preds[preds["model"] == "Main AI Model"].copy()
    sample = lgbm.sample(min(500, len(lgbm)), random_state=42) if not lgbm.empty else lgbm
    if not sample.empty:
        scatter = px.scatter(sample, x="demand", y="forecast", opacity=0.55, color_discrete_sequence=["#1f3a5f"])
        max_val = max(sample["demand"].max(), sample["forecast"].max())
        scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="#a64032", dash="dash"))
        scatter = style_plotly(scatter, 330)
        scatter.update_layout(showlegend=False, xaxis_title="Actual sales", yaxis_title="Forecast sales")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("How Close The Main Model Was To Reality")
        st.plotly_chart(scatter, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

if not by_sku.empty:
    focus = by_sku.pivot(index="sku", columns="model", values=metric).reset_index()
    focus = focus.rename(columns={"sku": "Product", **model_labels})
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader(f"Accuracy By Product ({metric_labels.get(metric, metric)})")
    st.dataframe(focus.sort_values(by=focus.columns[1], ascending=False), use_container_width=True, hide_index=True, height=360)
    st.markdown("</div>", unsafe_allow_html=True)
