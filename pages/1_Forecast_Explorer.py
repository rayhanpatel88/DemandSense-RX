"""Page 2 — Forecast Explorer."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.utils.config import load_config
from src.pipeline import run_pipeline

st.set_page_config(page_title="Forecast Explorer · DemandSense-RX",
                   page_icon="📈", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f1f35; }
[data-testid="stSidebar"] * { color: #cdd9e5 !important; }
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading pipeline…")
def get_data():
    return run_pipeline(load_config())


data = get_data()
raw_df = data["raw_df"]
test_df = data["test_df"]
future_df = data["future_df"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Forecast Explorer")
    st.divider()
    skus = sorted(raw_df["sku"].unique())
    selected_sku = st.selectbox("Select SKU", skus, index=0)
    horizon = st.radio("Forecast Horizon", ["7 days", "30 days"], index=1)
    show_intervals = st.checkbox("Show Prediction Intervals", value=True)
    history_days = st.slider("Historical Days Shown", 30, 365, 180)
    st.divider()
    st.page_link("streamlit_app.py", label="← Executive Overview")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title(f"📈 Forecast Explorer — {selected_sku}")
st.caption("Interactive time series with confidence intervals and uncertainty scores")
st.divider()

horizon_days = 7 if horizon == "7 days" else 30

# Filter data
sku_hist = raw_df[raw_df["sku"] == selected_sku].sort_values("date").tail(history_days)
sku_test = test_df[test_df["sku"] == selected_sku].sort_values("date")
sku_future = (future_df[future_df["sku"] == selected_sku]
              .sort_values("date")
              .head(horizon_days))

# ── Metrics row ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Avg Daily Demand", f"{sku_hist['demand'].mean():.1f} units")
with col2:
    st.metric(f"{horizon} Forecast Total", f"{int(sku_future['forecast'].sum()):,} units")
with col3:
    avg_reliability = sku_test["reliability_score"].mean() if "reliability_score" in sku_test.columns else 0
    st.metric("Avg Reliability", f"{avg_reliability:.1%}")
with col4:
    conf_mode = sku_test["confidence"].mode()[0] if "confidence" in sku_test.columns and len(sku_test) > 0 else "N/A"
    conf_colours = {"high": "normal", "medium": "off", "low": "inverse"}
    st.metric("Confidence Level", conf_mode.title())

st.divider()

# ── Main chart ────────────────────────────────────────────────────────────────
fig = go.Figure()

# Historical demand
fig.add_trace(go.Scatter(
    x=sku_hist["date"], y=sku_hist["demand"],
    name="Historical Demand", mode="lines",
    line=dict(color="#4a90d9", width=1.5),
    opacity=0.9,
))

# Test set predictions
if len(sku_test) > 0:
    fig.add_trace(go.Scatter(
        x=sku_test["date"], y=sku_test["forecast"],
        name="Test Forecast (LightGBM)", mode="lines",
        line=dict(color="#f4a261", width=2, dash="dot"),
    ))

# Future forecast
if len(sku_future) > 0:
    fig.add_trace(go.Scatter(
        x=sku_future["date"], y=sku_future["forecast"],
        name=f"Future Forecast ({horizon})", mode="lines+markers",
        line=dict(color="#e84393", width=2.5),
        marker=dict(size=5),
    ))

# Confidence intervals (test)
if show_intervals and len(sku_test) > 0 and "upper" in sku_test.columns:
    fig.add_trace(go.Scatter(
        x=pd.concat([sku_test["date"], sku_test["date"].iloc[::-1]]),
        y=pd.concat([sku_test["upper"], sku_test["lower"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(244,162,97,0.15)",
        line=dict(color="rgba(244,162,97,0)"),
        name="80% CI (test)", showlegend=True,
    ))

# Confidence intervals (future)
if show_intervals and len(sku_future) > 0 and "upper" in sku_future.columns:
    fig.add_trace(go.Scatter(
        x=pd.concat([sku_future["date"], sku_future["date"].iloc[::-1]]),
        y=pd.concat([sku_future["upper"], sku_future["lower"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(232,67,147,0.15)",
        line=dict(color="rgba(232,67,147,0)"),
        name="80% CI (future)", showlegend=True,
    ))

# Train/test divider
train_cutoff = data["train_cutoff"]
fig.add_vline(x=train_cutoff.isoformat() if hasattr(train_cutoff, "isoformat") else train_cutoff, line_dash="dash", line_color="#ffffff",
              annotation_text="Train | Test", annotation_font_color="#ffffff",
              annotation_position="top left")

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,25,45,0.8)",
    height=420,
    legend=dict(orientation="h", y=1.05, x=0),
    xaxis_title="Date",
    yaxis_title="Demand (units)",
    margin=dict(l=0, r=0, t=40, b=0),
)
st.plotly_chart(fig, use_container_width=True)

# ── Forecast table + Reliability ─────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"Forecast Table — Next {horizon_days} Days")
    tbl = sku_future[["date", "forecast", "lower", "upper"]].copy()
    tbl["date"] = tbl["date"].dt.strftime("%Y-%m-%d")
    tbl["forecast"] = tbl["forecast"].round(1)
    tbl["lower"] = tbl["lower"].round(1)
    tbl["upper"] = tbl["upper"].round(1)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

with col_right:
    st.subheader("Reliability Breakdown")
    if "confidence" in sku_test.columns and len(sku_test) > 0:
        conf_counts = sku_test["confidence"].value_counts().reset_index()
        conf_counts.columns = ["confidence", "days"]
        color_map = {"high": "#44bb44", "medium": "#ffd700", "low": "#ff4444"}
        import plotly.express as px
        fig_conf = px.bar(conf_counts, x="confidence", y="days",
                          color="confidence", color_discrete_map=color_map,
                          template="plotly_dark", height=260)
        fig_conf.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(15,25,45,0.8)",
                               showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                               xaxis_title="Confidence Level", yaxis_title="Days")
        st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.info("Reliability data not available for this SKU")
