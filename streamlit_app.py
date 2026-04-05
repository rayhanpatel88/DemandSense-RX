"""
DemandSense-RX — Streamlit entry point (Executive Overview page).
Run with: streamlit run streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config import load_config
from src.pipeline import run_pipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DemandSense-RX",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a8e 100%);
        border-radius: 12px;
        padding: 20px 24px;
        color: white;
        text-align: center;
    }
    .metric-card .label { font-size: 0.85rem; opacity: 0.85; margin-bottom: 6px; }
    .metric-card .value { font-size: 2rem; font-weight: 700; }
    .metric-card .delta { font-size: 0.8rem; opacity: 0.75; margin-top: 4px; }
    .risk-critical { color: #ff4444; font-weight: 700; }
    .risk-high { color: #ff8c00; font-weight: 600; }
    .risk-medium { color: #ffd700; }
    .risk-low { color: #44bb44; }
    [data-testid="stSidebar"] { background: #0f1f35; }
    [data-testid="stSidebar"] * { color: #cdd9e5 !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached pipeline ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Running forecasting pipeline…")
def get_data():
    config = load_config()
    return run_pipeline(config)


# ── Load data ─────────────────────────────────────────────────────────────────
data = get_data()
inventory_df = data["inventory_df"]
test_df = data["test_df"]
future_df = data["future_df"]
raw_df = data["raw_df"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/warehouse.png", width=60)
    st.title("DemandSense-RX")
    st.caption("Autonomous Demand Forecasting & Robotic Fulfilment")
    st.divider()
    st.markdown("**Navigation**")
    st.page_link("streamlit_app.py",           label="📊 Executive Overview",  icon="🏠")
    st.page_link("pages/1_Forecast_Explorer.py",    label="📈 Forecast Explorer")
    st.page_link("pages/2_Inventory_Decisions.py",  label="🏭 Inventory Decisions")
    st.page_link("pages/3_Robotics_Simulation.py",  label="🤖 Robotics Simulation")
    st.page_link("pages/4_Explainability.py",       label="🔍 Explainability")
    st.page_link("pages/5_Backtesting.py",          label="📉 Backtesting")
    st.page_link("pages/6_Scenario_Simulator.py",   label="🎛️ Scenario Simulator")
    st.divider()
    st.caption("Data: synthetic · Model: LightGBM")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Executive Overview")
st.caption("Real-time KPIs · Top risk signals · Demand summary")
st.divider()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
total_forecast = int(future_df["forecast"].sum())
stockout_count = int((inventory_df["stockout_risk"].isin(["critical", "high"])).sum())
accuracy_wape = data["backtest_results"]["summary"]
lgbm_wape = accuracy_wape["WAPE"]["LightGBM"] if "LightGBM" in accuracy_wape.index else 0
avg_accuracy = round(100 - lgbm_wape, 1)
reorder_count = int(inventory_df["reorder_needed"].sum())

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">📦 30-Day Forecast Demand</div>
      <div class="value">{total_forecast:,}</div>
      <div class="delta">across {raw_df['sku'].nunique()} SKUs</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">⚠️ Stockout Risk SKUs</div>
      <div class="value">{stockout_count}</div>
      <div class="delta">critical + high risk</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">🎯 Forecast Accuracy</div>
      <div class="value">{avg_accuracy}%</div>
      <div class="delta">LightGBM · rolling backtest</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">🔄 SKUs Needing Reorder</div>
      <div class="value">{reorder_count}</div>
      <div class="delta">below reorder point</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 2: Charts ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Total Demand: Historical vs Forecast")
    hist_agg = (raw_df.groupby("date")["demand"].sum().reset_index()
                .rename(columns={"demand": "value"}))
    hist_agg["type"] = "Historical"

    fcst_agg = (future_df.groupby("date")["forecast"].sum().reset_index()
                .rename(columns={"forecast": "value"}))
    fcst_agg["type"] = "Forecast"

    combined = pd.concat([hist_agg.tail(180), fcst_agg], ignore_index=True)
    fig = px.line(combined, x="date", y="value", color="type",
                  color_discrete_map={"Historical": "#4a90d9", "Forecast": "#f4a261"},
                  template="plotly_dark", height=320)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,45,0.8)",
        legend_title_text="", margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="", yaxis_title="Units",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Stockout Risk Distribution")
    risk_counts = (inventory_df["stockout_risk"]
                   .value_counts()
                   .reset_index()
                   .rename(columns={"index": "risk", "stockout_risk": "count"}))
    risk_counts.columns = ["risk", "count"]
    color_map = {"critical": "#ff4444", "high": "#ff8c00",
                 "medium": "#ffd700", "low": "#44bb44"}
    fig2 = px.pie(risk_counts, values="count", names="risk",
                  color="risk", color_discrete_map=color_map,
                  template="plotly_dark", height=320, hole=0.4)
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Risk Level",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 3: Top Risky SKUs ─────────────────────────────────────────────────────
st.subheader("⚠️ Top Risky SKUs")
risky = (inventory_df[inventory_df["stockout_risk"].isin(["critical", "high", "medium"])]
         .sort_values("days_to_stockout")
         .head(10)[["sku", "days_to_stockout", "current_stock", "reorder_point",
                     "forecast_30d_total", "stockout_risk", "reorder_needed"]])

def colour_risk(val):
    colours = {"critical": "background-color:#3d0000;color:#ff6666",
               "high": "background-color:#3d1a00;color:#ff9944",
               "medium": "background-color:#3d3400;color:#ffd700",
               "low": "background-color:#003d00;color:#88ff88"}
    return colours.get(val, "")

st.dataframe(
    risky.style.applymap(colour_risk, subset=["stockout_risk"]),
    use_container_width=True,
    hide_index=True,
)

# ── Row 4: Forecast by Category ───────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("30-Day Forecast by SKU")
    fcst_by_sku = (future_df.groupby("sku")["forecast"].sum()
                   .sort_values(ascending=False).head(15).reset_index())
    fig3 = px.bar(fcst_by_sku, x="sku", y="forecast",
                  color="forecast", color_continuous_scale="Blues",
                  template="plotly_dark", height=280)
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(15,25,45,0.8)",
                       showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                       xaxis_title="", yaxis_title="Forecast Units",
                       coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

with col_b:
    st.subheader("Days to Stockout Distribution")
    finite_days = inventory_df[inventory_df["days_to_stockout"] < 999]["days_to_stockout"]
    fig4 = px.histogram(finite_days, nbins=20, template="plotly_dark", height=280,
                        color_discrete_sequence=["#f4a261"])
    fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(15,25,45,0.8)",
                       margin=dict(l=0, r=0, t=10, b=0),
                       xaxis_title="Days to Stockout", yaxis_title="SKU Count")
    st.plotly_chart(fig4, use_container_width=True)
