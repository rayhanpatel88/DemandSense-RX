"""Page 6 — Backtesting."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config import load_config
from src.pipeline import run_pipeline

st.set_page_config(page_title="Backtesting · DemandSense-RX",
                   page_icon="📉", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f1f35; }
[data-testid="stSidebar"] * { color: #cdd9e5 !important; }
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading pipeline…")
def get_data():
    return run_pipeline(load_config())


data = get_data()
backtest = data["backtest_results"]
summary = backtest["summary"]
by_sku = backtest["by_sku"]
predictions = backtest["predictions"]

with st.sidebar:
    st.title("📉 Backtesting")
    st.divider()
    selected_metric = st.selectbox("Primary Metric", ["MAE", "RMSE", "MAPE", "WAPE"], index=3)
    st.divider()
    st.page_link("streamlit_app.py", label="← Executive Overview")

st.title("📉 Backtesting & Model Comparison")
st.caption("Rolling-origin evaluation · Multi-model comparison · Error analysis by SKU")
st.divider()

if len(summary) == 0:
    st.warning("No backtesting results available. Check that the pipeline ran successfully.")
    st.stop()

# ── Summary metrics table ─────────────────────────────────────────────────────
st.subheader("📊 Model Comparison Summary")
st.caption("Average metrics across all rolling folds")

def highlight_best(df):
    styled = df.copy().astype(str)
    for col in ["MAE", "RMSE", "MAPE", "WAPE"]:
        if col in df.columns:
            best_val = df[col].min()
            styled[col] = df[col].apply(
                lambda v: f"**{v:.3f}** ✓" if abs(v - best_val) < 0.001 else f"{v:.3f}"
            )
    return styled

col_summary, col_chart = st.columns([1, 2])
with col_summary:
    st.dataframe(summary.reset_index().rename(columns={"index": "model"}),
                 use_container_width=True, hide_index=True)

with col_chart:
    fig = go.Figure()
    metrics = ["MAE", "RMSE", "MAPE", "WAPE"]
    model_colors = {
        "LightGBM": "#4a90d9",
        "MovingAverage(w=7)": "#f4a261",
        "SeasonalNaive(s=7)": "#44bb44",
    }
    for model in summary.index:
        color = model_colors.get(model, "#aaaaaa")
        fig.add_trace(go.Bar(
            name=model,
            x=metrics,
            y=[summary.loc[model, m] for m in metrics if m in summary.columns],
            marker_color=color,
        ))
    fig.update_layout(
        barmode="group", template="plotly_dark", height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,45,0.8)",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Metric", yaxis_title="Value",
        legend=dict(orientation="h", y=1.08),
        title=f"All Metrics by Model",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Error over time (predictions df) ─────────────────────────────────────────
if len(predictions) > 0:
    st.subheader("📅 Forecast Error Over Time")
    predictions = predictions.copy()
    predictions["abs_error"] = (predictions["demand"] - predictions["forecast"]).abs()
    predictions["date"] = pd.to_datetime(predictions["date"])

    error_over_time = (predictions.groupby(["date", "model"])["abs_error"]
                       .mean().reset_index())
    fig2 = px.line(error_over_time, x="date", y="abs_error", color="model",
                   color_discrete_map=model_colors,
                   template="plotly_dark", height=300,
                   labels={"abs_error": "Mean Absolute Error", "date": "Date"})
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,45,0.8)",
        margin=dict(l=0, r=0, t=10, b=0), legend_title="Model",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

# ── Error by SKU ──────────────────────────────────────────────────────────────
if len(by_sku) > 0:
    st.subheader(f"🏷️ {selected_metric} by SKU")
    by_sku_pivot = by_sku.pivot(index="sku", columns="model", values=selected_metric)

    col_tbl, col_bar = st.columns([1, 2])
    with col_tbl:
        st.dataframe(by_sku_pivot.round(2), use_container_width=True)

    with col_bar:
        sku_top = by_sku_pivot.reset_index().melt(id_vars="sku",
                                                   var_name="model",
                                                   value_name=selected_metric)
        fig3 = px.bar(sku_top.sort_values(selected_metric, ascending=False).head(60),
                      x="sku", y=selected_metric, color="model",
                      color_discrete_map=model_colors,
                      barmode="group", template="plotly_dark", height=320,
                      labels={selected_metric: selected_metric, "sku": "SKU"})
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,45,0.8)",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_tickangle=-45,
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig3, use_container_width=True)

# ── Actual vs Predicted scatter ───────────────────────────────────────────────
if len(predictions) > 0:
    st.subheader("🎯 Actual vs Predicted Scatter")
    lgbm_preds = predictions[predictions["model"] == "LightGBM"].sample(
        min(500, len(predictions[predictions["model"] == "LightGBM"])), random_state=42
    )
    fig4 = px.scatter(lgbm_preds, x="demand", y="forecast",
                      opacity=0.5, template="plotly_dark", height=320,
                      color_discrete_sequence=["#4a90d9"],
                      labels={"demand": "Actual", "forecast": "Predicted"})
    max_val = max(lgbm_preds["demand"].max(), lgbm_preds["forecast"].max())
    fig4.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                   line=dict(color="white", dash="dash"))
    fig4.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,45,0.8)",
        margin=dict(l=0, r=0, t=10, b=0), title="LightGBM — Actual vs Predicted",
    )
    st.plotly_chart(fig4, use_container_width=True)
