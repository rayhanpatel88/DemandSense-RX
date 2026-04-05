"""Page 7 — Scenario Simulator."""

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
from src.recommendations.inventory import InventoryEngine
from src.simulation.simulator import WarehouseSimulator

st.set_page_config(page_title="Scenario Simulator · DemandSense-RX",
                   page_icon="🎛️", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f1f35; }
[data-testid="stSidebar"] * { color: #cdd9e5 !important; }
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading pipeline…")
def get_data():
    return run_pipeline(load_config())


data = get_data()
config = data["config"]
future_df = data["future_df"]
raw_df = data["raw_df"]
inventory_df = data["inventory_df"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎛️ Scenario Simulator")
    st.caption("Adjust levers and see downstream impact")
    st.divider()

    st.markdown("**Demand Scenario**")
    demand_multiplier = st.slider("Demand Multiplier", 0.5, 3.0, 1.0, 0.05,
                                  help="Scale forecast demand up or down")
    price_change_pct = st.slider("Price Change (%)", -30, 50, 0, 5,
                                 help="Simulate a price increase/decrease effect on demand")

    st.markdown("**Supply / Operations**")
    lead_time = st.slider("Lead Time (days)", 1, 30,
                          config["inventory"]["default_lead_time_days"])
    service_level = st.slider("Service Level", 0.80, 0.99, 0.95, 0.01)

    st.markdown("**Robotics**")
    n_robots = st.slider("Number of Robots", 1, 10,
                         config["simulation"]["n_robots"])
    sim_steps = st.slider("Simulation Steps", 50, 300, 100)

    run_btn = st.button("▶ Apply Scenario", type="primary", use_container_width=True)
    reset_btn = st.button("↺ Reset to Baseline", use_container_width=True)
    st.divider()
    st.page_link("streamlit_app.py", label="← Executive Overview")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🎛️ Scenario Simulator")
st.caption("What-if analysis: adjust demand, supply and robotics parameters dynamically")
st.divider()

# Price elasticity approximation: -0.5 elasticity
price_demand_adj = 1.0 + (price_change_pct / 100) * (-0.5)
effective_multiplier = demand_multiplier * max(price_demand_adj, 0.1)

# ── Compute scenario ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running scenario…")
def compute_scenario(effective_multiplier: float, lead_time: int,
                     service_level: float, n_robots: int, sim_steps: int):
    cfg = load_config()
    cfg["simulation"]["n_robots"] = n_robots
    cfg["simulation"]["time_steps"] = sim_steps

    # Adjusted forecasts
    fcst_adj = future_df.copy()
    fcst_adj["forecast"] = np.clip(fcst_adj["forecast"] * effective_multiplier, 0, None)
    fcst_adj["lower"] = np.clip(fcst_adj["lower"] * effective_multiplier, 0, None)
    fcst_adj["upper"] = np.clip(fcst_adj["upper"] * effective_multiplier, 0, None)

    # Inventory with new params
    inv_engine = InventoryEngine(cfg)
    inv_adj = inv_engine.compute(raw_df, fcst_adj, lead_time=lead_time,
                                 service_level=service_level)

    # Robot simulation
    sim = WarehouseSimulator(cfg)
    sim.run()
    sim_metrics = sim.get_metrics()

    return fcst_adj, inv_adj, sim_metrics


if run_btn or "scenario_results" not in st.session_state:
    with st.spinner("Computing scenario…"):
        fcst_adj, inv_adj, sim_metrics = compute_scenario(
            effective_multiplier, lead_time, service_level, n_robots, sim_steps
        )
    st.session_state["scenario_results"] = (fcst_adj, inv_adj, sim_metrics)

if reset_btn:
    if "scenario_results" in st.session_state:
        del st.session_state["scenario_results"]
    st.rerun()

fcst_adj, inv_adj, sim_metrics = st.session_state.get(
    "scenario_results",
    (future_df, inventory_df, {"fulfilment_rate": 0, "avg_robot_utilisation": 0,
                                "tasks_completed": 0, "avg_fulfilment_time": 0})
)

# ── Delta KPIs ────────────────────────────────────────────────────────────────
baseline_total = int(future_df["forecast"].sum())
scenario_total = int(fcst_adj["forecast"].sum())
delta_forecast = scenario_total - baseline_total

baseline_reorder = int(inventory_df["reorder_needed"].sum())
scenario_reorder = int(inv_adj["reorder_needed"].sum())
delta_reorder = scenario_reorder - baseline_reorder

baseline_stockout = int((inventory_df["stockout_risk"].isin(["critical", "high"])).sum())
scenario_stockout = int((inv_adj["stockout_risk"].isin(["critical", "high"])).sum())
delta_stockout = scenario_stockout - baseline_stockout

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Effective Demand Multiplier", f"{effective_multiplier:.2f}×")
c2.metric("30d Forecast Total", f"{scenario_total:,}",
          delta=f"{delta_forecast:+,} vs baseline")
c3.metric("SKUs Needing Reorder", scenario_reorder,
          delta=f"{delta_reorder:+} vs baseline")
c4.metric("High/Critical Risk SKUs", scenario_stockout,
          delta=f"{delta_stockout:+} vs baseline")
c5.metric("Robot Fulfilment Rate", f"{sim_metrics['fulfilment_rate']}%")

st.divider()

# ── Scenario vs Baseline Forecast ────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📈 Scenario vs Baseline — Total Forecast")
    base_agg = future_df.groupby("date")["forecast"].sum().reset_index()
    base_agg["series"] = "Baseline"
    scen_agg = fcst_adj.groupby("date")["forecast"].sum().reset_index()
    scen_agg["series"] = "Scenario"
    combined = pd.concat([base_agg, scen_agg], ignore_index=True)
    combined.rename(columns={"forecast": "value"}, inplace=True)

    fig = px.line(combined, x="date", y="value", color="series",
                  color_discrete_map={"Baseline": "#4a90d9", "Scenario": "#e84393"},
                  template="plotly_dark", height=300)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,45,0.8)",
        margin=dict(l=0, r=0, t=10, b=0), xaxis_title="", yaxis_title="Units",
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("🏭 Inventory Impact — Days to Stockout")
    compare_inv = pd.merge(
        inventory_df[["sku", "days_to_stockout"]].rename(columns={"days_to_stockout": "baseline"}),
        inv_adj[["sku", "days_to_stockout"]].rename(columns={"days_to_stockout": "scenario"}),
        on="sku"
    ).head(15)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Baseline", x=compare_inv["sku"],
                          y=compare_inv["baseline"], marker_color="#4a90d9"))
    fig2.add_trace(go.Bar(name="Scenario", x=compare_inv["sku"],
                          y=compare_inv["scenario"], marker_color="#e84393"))
    fig2.update_layout(
        barmode="group", template="plotly_dark", height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,45,0.8)",
        margin=dict(l=0, r=0, t=10, b=0), xaxis_title="", yaxis_title="Days",
        xaxis_tickangle=-45, legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Robot performance ─────────────────────────────────────────────────────────
st.subheader("🤖 Scenario Robot Performance")
col_r1, col_r2, col_r3, col_r4 = st.columns(4)
col_r1.metric("Tasks Completed", sim_metrics["tasks_completed"])
col_r2.metric("Tasks Pending", sim_metrics["tasks_pending"])
col_r3.metric("Avg Fulfilment Time", f"{sim_metrics['avg_fulfilment_time']} steps")
col_r4.metric("Avg Robot Utilisation", f"{sim_metrics['avg_robot_utilisation']}%")

# ── Scenario inventory table ──────────────────────────────────────────────────
st.divider()
st.subheader("📋 Scenario Inventory Decisions")
scenario_display = inv_adj[["sku", "mean_daily_demand", "safety_stock",
                             "reorder_point", "days_to_stockout",
                             "stockout_risk", "reorder_needed"]].copy()

risk_color_map = {
    "critical": "background-color:#3d0000;color:#ff6666",
    "high": "background-color:#3d1a00;color:#ff9944",
    "medium": "background-color:#3d3400;color:#ffd700",
    "low": "background-color:#003d00;color:#88ff88",
}

st.dataframe(
    scenario_display.style
    .applymap(lambda v: risk_color_map.get(v, ""), subset=["stockout_risk"]),
    use_container_width=True, hide_index=True, height=350,
)
