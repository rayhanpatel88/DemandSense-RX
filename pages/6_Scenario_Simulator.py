"""Scenario simulator page."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st

try:
    from src.app.ui import apply_page_config, get_pipeline_data, metric_panel, render_header, render_sidebar, style_plotly
except ImportError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.app.ui import apply_page_config, get_pipeline_data, metric_panel, render_header, render_sidebar, style_plotly
from src.recommendations.inventory import InventoryEngine
from src.simulation.simulator import WarehouseSimulator

apply_page_config("Scenario Simulator")
data = get_pipeline_data()
render_sidebar("scenario", data)

base_future = data["future_df"]
base_inventory = data["inventory_df"]
config = data["config"]

with st.sidebar:
    st.divider()
    demand_lift = st.slider("Demand multiplier", 0.6, 2.4, 1.0, step=0.05)
    promo_intensity = st.slider("Promo uplift", 0.8, 1.5, 1.0, step=0.05)
    lead_time = st.slider("Lead time", 2, 20, config["inventory"]["default_lead_time_days"])
    service_level = st.slider("Service level", 0.85, 0.99, config["inventory"]["default_service_level"], step=0.01)
    robots = st.slider("Robots", 1, 10, config["simulation"]["n_robots"])


@st.cache_data(show_spinner="Running scenario analysis...")
def simulate(demand_multiplier: float, promo_multiplier: float, lead_days: int, service_target: float, robot_count: int):
    scenario_future = base_future.copy()
    scenario_future["forecast"] = scenario_future["forecast"] * demand_multiplier * np.where(scenario_future["promotion"] > 0, promo_multiplier, 1.0)
    scenario_future["lower"] = scenario_future["lower"] * demand_multiplier
    scenario_future["upper"] = scenario_future["upper"] * demand_multiplier * np.where(scenario_future["promotion"] > 0, promo_multiplier, 1.0)
    inventory = InventoryEngine(config).compute(data["raw_df"], scenario_future, lead_time=lead_days, service_level=service_target)
    scenario_config = dict(config)
    scenario_config["simulation"] = dict(config["simulation"])
    scenario_config["simulation"]["n_robots"] = robot_count
    simulation = WarehouseSimulator(scenario_config, forecast_df=scenario_future, inventory_df=inventory, slotting_df=data["slotting_df"]).run()
    return scenario_future, inventory, simulation.get_metrics()


scenario_future, scenario_inventory, sim_metrics = simulate(demand_lift, promo_intensity, lead_time, service_level, robots)

render_header(
    "Scenario Planning",
    "Scenario Simulator",
    "Stress demand, lead time, and labor capacity on the left; inspect resulting forecast volume, inventory posture, and warehouse throughput on the right.",
)

for col, html in zip(
    st.columns(4),
    [
        metric_panel("Scenario Demand", f"{scenario_future['forecast'].sum():,.0f}", f"{scenario_future['forecast'].sum() - base_future['forecast'].sum():+,.0f} units vs baseline"),
        metric_panel("Reorder SKUs", f"{int(scenario_inventory['reorder_needed'].sum())}", f"{int(scenario_inventory['reorder_needed'].sum() - base_inventory['reorder_needed'].sum()):+d} vs baseline"),
        metric_panel("Execution Rate", f"{sim_metrics['fulfilment_rate']:.1f}%", "Warehouse completion rate"),
        metric_panel("Delayed Picks", f"{sim_metrics['inventory_linked_delays']}", "Shortage-linked execution friction"),
    ],
):
    with col:
        st.markdown(html, unsafe_allow_html=True)

controls, outputs = st.columns([0.9, 1.5], gap="large")
with controls:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Scenario Readout")
    st.markdown(
        f"<div class='note'>Demand multiplier {demand_lift:.2f}x with promo factor {promo_intensity:.2f}x. "
        f"Lead time is {lead_time} days and service target is {service_level:.1%}. "
        f"Warehouse labor is constrained to {robots} robots.</div>",
        unsafe_allow_html=True,
    )
    focus = scenario_inventory.sort_values(["reorder_needed", "days_to_stockout"], ascending=[False, True]).head(6)
    for _, row in focus.iterrows():
        st.markdown(f"<div class='note'><span class='tag'>{row['sku']}</span>{row['explanation']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with outputs:
    baseline = base_future.groupby("date")["forecast"].sum().reset_index(name="baseline")
    scenario = scenario_future.groupby("date")["forecast"].sum().reset_index(name="scenario")
    merged = baseline.merge(scenario, on="date")
    lines = px.line(merged.melt(id_vars="date", var_name="series", value_name="forecast"), x="date", y="forecast", color="series", color_discrete_map={"baseline": "#6d685f", "scenario": "#1f3a5f"})
    lines = style_plotly(lines, 320)
    lines.update_layout(legend=dict(orientation="h", y=1.08, x=0), xaxis_title="", yaxis_title="Units")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Demand Delta")
    st.plotly_chart(lines, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    compare = scenario_inventory[["sku", "days_to_stockout"]].merge(base_inventory[["sku", "days_to_stockout"]], on="sku", suffixes=("_scenario", "_baseline"))
    top = compare.sort_values("days_to_stockout_scenario").head(12)
    bars = px.bar(top.melt(id_vars="sku", var_name="series", value_name="days"), x="sku", y="days", color="series", barmode="group", color_discrete_map={"days_to_stockout_scenario": "#1f3a5f", "days_to_stockout_baseline": "#b4a06a"})
    bars = style_plotly(bars, 320)
    bars.update_layout(legend=dict(orientation="h", y=1.08, x=0), xaxis_title="", yaxis_title="Days to stockout")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Inventory Impact")
    st.plotly_chart(bars, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)
