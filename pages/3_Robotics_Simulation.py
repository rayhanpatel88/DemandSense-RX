"""Warehouse robotics page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app.ui import apply_page_config, get_pipeline_data, metric_panel, render_header, render_sidebar, style_plotly
from src.simulation.robot import RobotStatus
from src.simulation.simulator import WarehouseSimulator
from src.simulation.warehouse import Cell

apply_page_config("Warehouse Robotics")
data = get_pipeline_data()
render_sidebar("robotics", data)

with st.sidebar:
    st.divider()
    n_robots = st.slider("Robots", 1, 8, data["config"]["simulation"]["n_robots"])
    steps = st.slider("Simulation steps", 60, 240, data["config"]["simulation"]["time_steps"], step=20)
    load_factor = st.slider("Load factor", 0.2, 1.6, 1.0, step=0.1)
    rerun = st.button("Run simulation", use_container_width=True, type="primary")


@st.cache_data(show_spinner="Simulating warehouse execution...")
def run_simulation(robot_count: int, sim_steps: int, factor: float):
    config = data["config"].copy()
    config["simulation"] = dict(config["simulation"])
    config["simulation"]["n_robots"] = robot_count
    config["simulation"]["time_steps"] = sim_steps
    config["simulation"]["orders_per_step"] = config["simulation"]["orders_per_step"] * factor
    simulator = WarehouseSimulator(
        config,
        forecast_df=data["future_df"],
        inventory_df=data["inventory_df"],
        slotting_df=data["slotting_df"],
    ).run()
    return {
        "metrics": simulator.get_metrics(),
        "paths": simulator.get_robot_paths_df(),
        "congestion": simulator.get_congestion_df(),
        "tasks": simulator.get_task_log(),
        "grid": simulator.warehouse.to_numpy(),
        "history": simulator.history,
    }


if rerun or "robotics_result" not in st.session_state:
    st.session_state["robotics_result"] = run_simulation(n_robots, steps, load_factor)

result = st.session_state["robotics_result"]
metrics = result["metrics"]
tasks = result["tasks"]
congestion = result["congestion"]
grid = result["grid"]
history = result["history"]

render_header(
    "Execution Simulation",
    "Warehouse Robotics",
    "Pick tasks are now generated from forecast mix and slotted SKU locations, so congestion and delay are consequences of demand pressure rather than random task spam.",
)

for col, html in zip(
    st.columns(5),
    [
        metric_panel("Tasks Generated", f"{metrics['tasks_generated']}", "Forecast-weighted picks created"),
        metric_panel("Fulfilment Rate", f"{metrics['fulfilment_rate']:.1f}%", "Completed versus generated"),
        metric_panel("Avg Time", f"{metrics['avg_fulfilment_time']:.1f}", "Steps per completed task"),
        metric_panel("Queue Delay", f"{metrics['avg_queue_delay']:.1f}", "Waiting time before assignment"),
        metric_panel("Inventory Delays", f"{metrics['inventory_linked_delays']}", "Tasks slowed by shortage pressure"),
    ],
):
    with col:
        st.markdown(html, unsafe_allow_html=True)

left, right = st.columns([1.55, 1.0], gap="large")
with left:
    step = st.slider("Playback step", 0, max(len(history) - 1, 0), min(20, max(len(history) - 1, 0)))
    fig = go.Figure()
    background = np.where(grid == Cell.SHELF, 0.65, np.where(grid == Cell.PACKING, 0.25, 0.05))
    fig.add_trace(go.Heatmap(z=background, colorscale=[[0, "#f2eee7"], [0.45, "#c8d5e2"], [1, "#495e76"]], showscale=False))
    color_map = {
        RobotStatus.IDLE.name: "#5a715f",
        RobotStatus.MOVING_TO_SHELF.name: "#1f3a5f",
        RobotStatus.PICKING.name: "#8c5a2b",
        RobotStatus.MOVING_TO_PACKING.name: "#6d685f",
        RobotStatus.DELIVERING.name: "#a64032",
        RobotStatus.DELAYED.name: "#ad7b2b",
    }
    if history:
        for snap in history[step]:
            fig.add_trace(
                go.Scatter(
                    x=[snap["col"]],
                    y=[snap["row"]],
                    mode="markers+text",
                    text=[f"R{snap['robot_id']}"],
                    textposition="middle center",
                    marker=dict(size=18, color=color_map.get(snap["status"], "#161616"), line=dict(color="#faf8f4", width=1.4)),
                    name=snap["status"].replace("_", " ").title(),
                    showlegend=False,
                )
            )
    fig = style_plotly(fig, 470)
    fig.update_layout(yaxis=dict(scaleanchor="x", autorange="reversed", showticklabels=False), xaxis=dict(showticklabels=False))
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Live Warehouse State")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not tasks.empty:
        throughput = tasks.groupby("zone").size().rename("tasks").reset_index()
        zone = px.bar(throughput, x="zone", y="tasks", color="zone", color_discrete_map={"A": "#1f3a5f", "B": "#8c5a2b", "C": "#6d685f"})
        zone = style_plotly(zone, 280)
        zone.update_layout(showlegend=False, xaxis_title="Zone", yaxis_title="Tasks")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Zone Traffic Mix")
        st.plotly_chart(zone, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    heat = congestion.pivot(index="row", columns="col", values="visits").fillna(0)
    heatmap = px.imshow(heat, color_continuous_scale=["#f5f0e8", "#d9c9ae", "#8c5a2b", "#a64032"], aspect="equal")
    heatmap = style_plotly(heatmap, 300)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Congestion Heatmap")
    st.plotly_chart(heatmap, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Operational Notes")
    st.markdown(
        f"<div class='note'>High-demand zones: {metrics['zone_task_mix']}. "
        f"Reducing robots increases queue delay and average fulfilment time; shortage-linked picks add recovery delay before packing.</div>",
        unsafe_allow_html=True,
    )
    if not tasks.empty:
        focus = tasks.sort_values(["shortage_delay", "queue_delay"], ascending=[False, False]).head(5)
        for _, row in focus.iterrows():
            st.markdown(
                f"<div class='note'><span class='tag'>{row['sku']}</span>Zone {row['zone']} | queue {row['queue_delay']} | shortage delay {row['shortage_delay']}</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)
