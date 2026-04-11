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

# ---------------------------------------------------------------------------
# Named scenario presets — modelled on real pharmaceutical DC events
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    "Baseline": {
        "demand_lift": 1.0, "promo_intensity": 1.0,
        "lead_time": config["inventory"]["default_lead_time_days"],
        "service_level": config["inventory"]["default_service_level"],
        "robots": config["simulation"]["n_robots"],
        "description": "Current operating plan with no adjustments.",
    },
    "Flu Season Surge": {
        "demand_lift": 1.55, "promo_intensity": 1.20,
        "lead_time": config["inventory"]["default_lead_time_days"],
        "service_level": 0.98,
        "robots": config["simulation"]["n_robots"] + 2,
        "description": (
            "Oct–Feb cold/flu peak. OTC cold remedies and antibiotics spike 40–60%. "
            "Modelled on 2022–23 RSV/flu co-circulation season."
        ),
    },
    "Allergy Season": {
        "demand_lift": 1.35, "promo_intensity": 1.15,
        "lead_time": config["inventory"]["default_lead_time_days"],
        "service_level": 0.97,
        "robots": config["simulation"]["n_robots"] + 1,
        "description": (
            "Apr–Jun tree/grass pollen peak. Antihistamine demand rises 30–45% "
            "across the OTC allergy category."
        ),
    },
    "Supply Chain Disruption": {
        "demand_lift": 1.10, "promo_intensity": 1.0,
        "lead_time": min(config["inventory"]["default_lead_time_days"] + 8, 21),
        "service_level": 0.95,
        "robots": config["simulation"]["n_robots"],
        "description": (
            "Supplier delay of +8 days (port congestion / force-majeure event). "
            "Tests how quickly stockout exposure accumulates under extended lead times."
        ),
    },
    "GPO Contract Win": {
        "demand_lift": 1.45, "promo_intensity": 1.30,
        "lead_time": config["inventory"]["default_lead_time_days"],
        "service_level": 0.98,
        "robots": config["simulation"]["n_robots"] + 2,
        "description": (
            "New Group Purchasing Organisation contract adds 8–12 new hospital accounts. "
            "Volume ramp starts at 1.45× with promotional price commitments."
        ),
    },
    "Pandemic Demand Spike": {
        "demand_lift": 2.10, "promo_intensity": 1.0,
        "lead_time": min(config["inventory"]["default_lead_time_days"] + 5, 21),
        "service_level": 0.99,
        "robots": min(config["simulation"]["n_robots"] + 4, 10),
        "description": (
            "Extreme demand surge (2.1×) with concurrent lead-time pressure (+5d). "
            "Models a public-health emergency activation such as a novel pathogen event."
        ),
    },
    "Year-End Budget Flush": {
        "demand_lift": 1.25, "promo_intensity": 1.10,
        "lead_time": config["inventory"]["default_lead_time_days"],
        "service_level": 0.96,
        "robots": config["simulation"]["n_robots"],
        "description": (
            "Q4 hospital budget-year end: departments deplete remaining PO budgets, "
            "generating a 20–30% demand uptick across surgical and Rx categories."
        ),
    },
    "Stockout Recovery": {
        "demand_lift": 0.85, "promo_intensity": 0.95,
        "lead_time": max(config["inventory"]["default_lead_time_days"] - 2, 2),
        "service_level": 0.99,
        "robots": config["simulation"]["n_robots"] + 1,
        "description": (
            "Post-disruption restocking phase. Demand dips slightly as backorders clear "
            "but service-level target is elevated to rebuild safety stock."
        ),
    },
}

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.divider()
    st.markdown(
        "<div style='font-size:0.67rem;font-weight:700;letter-spacing:0.18em;"
        "text-transform:uppercase;color:var(--muted);margin-bottom:0.5rem;'>Scenario Preset</div>",
        unsafe_allow_html=True,
    )
    selected_preset = st.selectbox(
        "Scenario",
        list(PRESETS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    preset = PRESETS[selected_preset]

    st.markdown(
        f"<div class='note' style='margin-bottom:0.75rem;font-size:0.8rem;'>{preset['description']}</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        "<div style='font-size:0.67rem;font-weight:700;letter-spacing:0.18em;"
        "text-transform:uppercase;color:var(--muted);margin-bottom:0.5rem;'>Fine-Tune</div>",
        unsafe_allow_html=True,
    )
    demand_lift = st.slider(
        "Demand multiplier", 0.6, 2.4, float(preset["demand_lift"]), step=0.05,
        key=f"dm_{selected_preset}",
    )
    promo_intensity = st.slider(
        "Promo uplift", 0.8, 1.5, float(preset["promo_intensity"]), step=0.05,
        key=f"pi_{selected_preset}",
    )
    lead_time = st.slider(
        "Lead time (days)", 2, 21, int(preset["lead_time"]),
        key=f"lt_{selected_preset}",
    )
    service_level = st.slider(
        "Service level", 0.85, 0.99, float(preset["service_level"]), step=0.01,
        key=f"sl_{selected_preset}",
    )
    robots = st.slider(
        "Robots", 1, 10, int(preset["robots"]),
        key=f"rb_{selected_preset}",
    )


@st.cache_data(show_spinner="Running scenario analysis...")
def simulate(
    demand_multiplier: float,
    promo_multiplier: float,
    lead_days: int,
    service_target: float,
    robot_count: int,
):
    scenario_future = base_future.copy()
    promo_mask = np.where(scenario_future.get("promotion", 0) > 0, promo_multiplier, 1.0)
    scenario_future["forecast"] = scenario_future["forecast"] * demand_multiplier * promo_mask
    scenario_future["lower"] = scenario_future["lower"] * demand_multiplier
    scenario_future["upper"] = scenario_future["upper"] * demand_multiplier * promo_mask
    inventory = InventoryEngine(config).compute(
        data["raw_df"], scenario_future,
        lead_time=lead_days, service_level=service_target,
    )
    scenario_config = dict(config)
    scenario_config["simulation"] = dict(config["simulation"])
    scenario_config["simulation"]["n_robots"] = robot_count
    simulation = WarehouseSimulator(
        scenario_config,
        forecast_df=scenario_future,
        inventory_df=inventory,
        slotting_df=data["slotting_df"],
    ).run()
    return scenario_future, inventory, simulation.get_metrics()


scenario_future, scenario_inventory, sim_metrics = simulate(
    demand_lift, promo_intensity, lead_time, service_level, robots
)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
render_header(
    "Scenario Planning",
    "Scenario Simulator",
    "Change demand, lead time, service level, or robot capacity to see how the business plan would shift under a different scenario.",
)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
demand_delta = scenario_future["forecast"].sum() - base_future["forecast"].sum()
reorder_delta = int(scenario_inventory["reorder_needed"].sum() - base_inventory["reorder_needed"].sum())

for col, html in zip(
    st.columns(4),
    [
        metric_panel(
            "Expected Sales In This Scenario",
            f"{scenario_future['forecast'].sum():,.0f}",
            f"{demand_delta:+,.0f} units vs baseline",
        ),
        metric_panel(
            "Products To Reorder",
            f"{int(scenario_inventory['reorder_needed'].sum())}",
            f"{reorder_delta:+d} vs baseline",
        ),
        metric_panel(
            "Warehouse Completion Rate",
            f"{sim_metrics['fulfilment_rate']:.1f}%",
            "Warehouse order completion rate",
        ),
        metric_panel(
            "Delays From Shortages",
            f"{sim_metrics['inventory_linked_delays']}",
            "Picks delayed by stock shortage",
        ),
    ],
):
    with col:
        st.markdown(html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
controls, outputs = st.columns([0.9, 1.5], gap="large")

with controls:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader(f"Scenario Summary: {selected_preset}")
    st.markdown(
        f"<div class='note'>{preset['description']}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='note' style='margin-top:0.6rem;'>"
        f"Demand <strong>{demand_lift:.2f}×</strong> · "
        f"Promotion lift <strong>{promo_intensity:.2f}×</strong> · "
        f"Lead time <strong>{lead_time}d</strong> · "
        f"Service target <strong>{service_level:.1%}</strong> · "
        f"<strong>{robots}</strong> robots"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Scenario comparison table
    all_presets_summary = []
    for pname, pvals in PRESETS.items():
        all_presets_summary.append({
            "Scenario": pname,
            "Demand Change": f"{pvals['demand_lift']:.2f}x",
            "Lead Time (Days)": pvals["lead_time"],
            "Service Target": f"{pvals['service_level']:.0%}",
            "Robots": pvals["robots"],
        })
    import pandas as _pd
    st.markdown(
        "<div style='margin-top:0.9rem;font-size:0.67rem;font-weight:700;"
        "letter-spacing:0.14em;text-transform:uppercase;color:var(--muted);"
        "margin-bottom:0.4rem;'>All Presets</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        _pd.DataFrame(all_presets_summary),
        hide_index=True,
        height=260,
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Recommended Actions")
    focus = scenario_inventory.sort_values(
        ["reorder_needed", "days_to_stockout"], ascending=[False, True]
    ).head(6)
    for _, row in focus.iterrows():
        st.markdown(
            f"<div class='note' style='margin-bottom:0.6rem;'>"
            f"<span class='tag'>{row['sku']}</span>{row['explanation']}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with outputs:
    baseline_agg = base_future.groupby("date")["forecast"].sum().reset_index(name="Baseline")
    scenario_agg = scenario_future.groupby("date")["forecast"].sum().reset_index(name=selected_preset)
    merged = baseline_agg.merge(scenario_agg, on="date")
    lines = px.line(
        merged.melt(id_vars="date", var_name="Series", value_name="Units"),
        x="date", y="Units", color="Series",
        color_discrete_map={"Baseline": "#4b5563", selected_preset: "#2563eb"},
    )
    lines = style_plotly(lines, 320)
    lines.for_each_trace(
        lambda trace: trace.update(
            line=dict(width=3.0 if trace.name == selected_preset else 1.8, dash="solid" if trace.name == selected_preset else "dot"),
            opacity=1.0 if trace.name == selected_preset else 0.65,
        )
    )
    lines.update_layout(
        xaxis_title="", yaxis_title="Expected units sold",
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Expected Sales: Scenario vs Current Plan")
    st.caption("This compares expected sales in the scenario against the current plan over the next 30 days.")
    st.plotly_chart(lines, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    compare = scenario_inventory[["sku", "days_to_stockout"]].merge(
        base_inventory[["sku", "days_to_stockout"]], on="sku",
        suffixes=("_scenario", "_baseline"),
    )
    top = compare.sort_values("days_to_stockout_scenario").head(12)
    bars = px.bar(
        top.melt(id_vars="sku", var_name="Series", value_name="Days"),
        x="sku", y="Days", color="Series",
        barmode="group",
        color_discrete_map={
            "days_to_stockout_scenario": "#2563eb",
            "days_to_stockout_baseline": "#94a3b8",
        },
    )
    bars.for_each_trace(
        lambda trace: trace.update(
            name="Scenario plan" if trace.name == "days_to_stockout_scenario" else "Current plan",
            opacity=1.0 if trace.name == "days_to_stockout_scenario" else 0.55,
        )
    )
    bars = style_plotly(bars, 300)
    bars.update_layout(
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis_title="", yaxis_title="Days until stockout",
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("How Stock Cover Changes")
    st.caption("This shows which products would run out sooner or later under the scenario compared with today’s plan.")
    st.plotly_chart(bars, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Fulfilment rate vs robots sensitivity (static insight)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Warehouse Impact Summary")
    exec_items = [
        ("Warehouse completion rate", f"{sim_metrics['fulfilment_rate']:.1f}%"),
        ("Delays caused by shortages", f"{sim_metrics['inventory_linked_delays']}"),
        ("Active robots", f"{robots}"),
        ("Lead time used", f"{lead_time}d"),
        ("Service target", f"{service_level:.1%}"),
        ("Demand change", f"{demand_lift:.2f}×"),
    ]
    st.markdown(
        "".join([
            f"<div class='note' style='display:flex;justify-content:space-between;"
            f"gap:0.75rem;padding:0.65rem 0;border-bottom:1px solid var(--line);'>"
            f"<span>{label}</span><strong style='color:var(--text);'>{value}</strong></div>"
            for label, value in exec_items
        ]),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
