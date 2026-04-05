"""Page 3 — Inventory Decisions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config import load_config
from src.pipeline import run_pipeline

st.set_page_config(page_title="Inventory Decisions · DemandSense-RX",
                   page_icon="🏭", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f1f35; }
[data-testid="stSidebar"] * { color: #cdd9e5 !important; }
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading pipeline…")
def get_data():
    return run_pipeline(load_config())


data = get_data()
inventory_df = data["inventory_df"]
stockout_timeline = data["stockout_timeline"]
raw_df = data["raw_df"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏭 Inventory Decisions")
    st.divider()
    risk_filter = st.multiselect(
        "Filter by Risk Level",
        ["critical", "high", "medium", "low"],
        default=["critical", "high", "medium", "low"],
    )
    show_reorder_only = st.checkbox("Show Reorder Needed Only", value=False)
    st.divider()
    st.page_link("streamlit_app.py", label="← Executive Overview")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🏭 Inventory Decisions")
st.caption("Reorder recommendations · Safety stock · Stockout timeline")
st.divider()

# KPI row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total SKUs", len(inventory_df))
with c2:
    critical = int((inventory_df["stockout_risk"] == "critical").sum())
    st.metric("Critical Risk", critical, delta=None)
with c3:
    needs_reorder = int(inventory_df["reorder_needed"].sum())
    st.metric("Reorder Needed", needs_reorder)
with c4:
    avg_days = inventory_df[inventory_df["days_to_stockout"] < 999]["days_to_stockout"].mean()
    st.metric("Avg Days to Stockout", f"{avg_days:.1f}")

st.divider()

# ── Reorder Table ─────────────────────────────────────────────────────────────
st.subheader("📋 Reorder Recommendations")

filtered = inventory_df[inventory_df["stockout_risk"].isin(risk_filter)].copy()
if show_reorder_only:
    filtered = filtered[filtered["reorder_needed"] == 1]

display_cols = ["sku", "mean_daily_demand", "current_stock", "safety_stock",
                "reorder_point", "reorder_qty", "days_to_stockout",
                "forecast_30d_total", "stockout_risk", "reorder_needed"]

risk_color_map = {
    "critical": "background-color:#3d0000;color:#ff6666",
    "high": "background-color:#3d1a00;color:#ff9944",
    "medium": "background-color:#3d3400;color:#ffd700",
    "low": "background-color:#003d00;color:#88ff88",
}


def style_risk(val):
    return risk_color_map.get(val, "")


def style_reorder(val):
    return "background-color:#3d0000;color:#ff6666;font-weight:700" if val == 1 else ""


styled = (
    filtered[display_cols]
    .style
    .map(style_risk, subset=["stockout_risk"])
    .map(style_reorder, subset=["reorder_needed"])
    .format({"mean_daily_demand": "{:.1f}", "days_to_stockout": "{:.1f}",
             "current_stock": "{:.0f}", "safety_stock": "{:.0f}",
             "reorder_point": "{:.0f}", "reorder_qty": "{:.0f}",
             "forecast_30d_total": "{:.0f}"})
)
st.dataframe(styled, use_container_width=True, hide_index=True, height=350)

# ── Row 2 ─────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🛡️ Safety Stock vs Reorder Point")
    top15 = inventory_df.nsmallest(15, "days_to_stockout")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Safety Stock", x=top15["sku"],
                         y=top15["safety_stock"], marker_color="#4a90d9"))
    fig.add_trace(go.Bar(name="Reorder Point", x=top15["sku"],
                         y=top15["reorder_point"], marker_color="#f4a261"))
    fig.update_layout(
        barmode="group", template="plotly_dark", height=320,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,45,0.8)",
        margin=dict(l=0, r=0, t=10, b=0), xaxis_title="", yaxis_title="Units",
        legend=dict(orientation="h", y=1.08),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("📦 Current Stock vs Forecast Demand")
    top15 = inventory_df.nsmallest(15, "days_to_stockout")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Current Stock", x=top15["sku"],
                          y=top15["current_stock"], marker_color="#44bb44"))
    fig2.add_trace(go.Bar(name="30d Forecast", x=top15["sku"],
                          y=top15["forecast_30d_total"], marker_color="#e84393"))
    fig2.update_layout(
        barmode="group", template="plotly_dark", height=320,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,45,0.8)",
        margin=dict(l=0, r=0, t=10, b=0), xaxis_title="", yaxis_title="Units",
        legend=dict(orientation="h", y=1.08),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Stockout Timeline ─────────────────────────────────────────────────────────
st.subheader("⏳ Projected Stock Timeline (Top 5 Riskiest SKUs)")
risky_skus = inventory_df.nsmallest(5, "days_to_stockout")["sku"].tolist()
timeline_filtered = stockout_timeline[stockout_timeline["sku"].isin(risky_skus)]

fig3 = px.line(timeline_filtered, x="date", y="projected_stock", color="sku",
               template="plotly_dark", height=320)
# Add reorder point lines
for _, row in inventory_df[inventory_df["sku"].isin(risky_skus)].iterrows():
    fig3.add_hline(y=row["reorder_point"], line_dash="dash", line_color="#ff8c00",
                   annotation_text=f"ROP: {row['sku']}", annotation_font_size=9)
fig3.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,45,0.8)",
    margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Date",
    yaxis_title="Projected Stock (units)", legend_title="SKU",
)
st.plotly_chart(fig3, use_container_width=True)
