"""Page 4 — Robotics Simulation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config import load_config
from src.simulation.simulator import WarehouseSimulator
from src.simulation.warehouse import Cell

st.set_page_config(page_title="Robotics Simulation · DemandSense-RX",
                   page_icon="🤖", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f1f35; }
[data-testid="stSidebar"] * { color: #cdd9e5 !important; }
</style>""", unsafe_allow_html=True)


# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Robotics Simulation")
    st.divider()
    n_robots = st.slider("Number of Robots", 1, 8, 3)
    time_steps = st.slider("Simulation Steps", 20, 200, 100)
    orders_per_step = st.slider("Order Arrival Rate", 0.1, 1.0, 0.3, 0.05)
    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)
    st.divider()
    st.page_link("streamlit_app.py", label="← Executive Overview")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🤖 Warehouse Robotics Simulation")
st.caption("Multi-agent A* pathfinding · Task assignment · Fulfilment metrics")
st.divider()

# ── Run simulation ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running simulation…")
def run_sim(n_robots: int, time_steps: int, orders_per_step: float):
    config = load_config()
    config["simulation"]["n_robots"] = n_robots
    config["simulation"]["time_steps"] = time_steps
    config["simulation"]["orders_per_step"] = orders_per_step
    sim = WarehouseSimulator(config)
    sim.run()
    metrics = sim.get_metrics()
    paths_df = sim.get_robot_paths_df()
    congestion_df = sim.get_congestion_df()
    warehouse_grid = sim.warehouse.to_numpy()
    return metrics, paths_df, congestion_df, warehouse_grid, sim.history


if "sim_results" not in st.session_state or run_btn:
    with st.spinner("Running warehouse simulation…"):
        results = run_sim(n_robots, time_steps, orders_per_step)
    st.session_state["sim_results"] = results

metrics, paths_df, congestion_df, warehouse_grid, history = st.session_state["sim_results"]

# ── Metrics ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tasks Generated", metrics["tasks_generated"])
c2.metric("Tasks Completed", metrics["tasks_completed"])
c3.metric("Fulfilment Rate", f"{metrics['fulfilment_rate']}%")
c4.metric("Avg Fulfilment Time", f"{metrics['avg_fulfilment_time']} steps")
c5.metric("Robot Utilisation", f"{metrics['avg_robot_utilisation']}%")

st.divider()

# ── Layout: Grid viz + Playback ───────────────────────────────────────────────
col_viz, col_ctrl = st.columns([3, 1])

with col_ctrl:
    st.subheader("Playback")
    step = st.slider("Step", 0, max(len(history) - 1, 1), 0,
                     key="step_slider")
    st.caption(f"Step {step} / {len(history) - 1}")
    auto_play = st.checkbox("Auto-advance (refresh page)")

with col_viz:
    st.subheader("Warehouse Grid & Robot Positions")

    # Build heatmap from warehouse grid
    H, W = warehouse_grid.shape
    cell_labels = {int(Cell.EMPTY): "Empty", int(Cell.SHELF): "Shelf",
                   int(Cell.AISLE): "Aisle", int(Cell.PACKING): "Packing"}

    # Background grid
    grid_colors = np.zeros((H, W), dtype=float)
    for r in range(H):
        for c in range(W):
            v = warehouse_grid[r, c]
            if v == Cell.SHELF:
                grid_colors[r, c] = 0.3
            elif v == Cell.PACKING:
                grid_colors[r, c] = 0.7
            else:
                grid_colors[r, c] = 0.0

    fig = go.Figure()

    # Warehouse background
    fig.add_trace(go.Heatmap(
        z=grid_colors,
        colorscale=[[0, "#0a1a2e"], [0.3, "#1a3a5e"], [0.7, "#2e5e3e"], [1.0, "#1a3a5e"]],
        showscale=False,
        zmin=0, zmax=1,
        hoverinfo="skip",
    ))

    # Draw shelves as shapes
    for r in range(H):
        for c in range(W):
            if warehouse_grid[r, c] == Cell.SHELF:
                fig.add_shape(type="rect",
                              x0=c - 0.45, x1=c + 0.45,
                              y0=r - 0.45, y1=r + 0.45,
                              fillcolor="#2d5a8e", line_color="#4a90d9", line_width=0.5)
            elif warehouse_grid[r, c] == Cell.PACKING:
                fig.add_shape(type="rect",
                              x0=c - 0.45, x1=c + 0.45,
                              y0=r - 0.45, y1=r + 0.45,
                              fillcolor="#2e5e3e", line_color="#44bb44", line_width=1)

    # Robot positions at current step
    if step < len(history):
        snap = history[step]
        robot_colors = ["#ff4444", "#44bbff", "#ffaa00", "#aa44ff",
                        "#ff44aa", "#44ffaa", "#ffff44", "#ff8844"]
        status_symbols = {
            "IDLE": "circle", "MOVING_TO_SHELF": "arrow-up",
            "PICKING": "star", "MOVING_TO_PACKING": "arrow-down",
            "DELIVERING": "diamond",
        }
        for s in snap:
            rid = s["robot_id"]
            color = robot_colors[rid % len(robot_colors)]
            symbol = status_symbols.get(s["status"], "circle")
            fig.add_trace(go.Scatter(
                x=[s["col"]], y=[s["row"]],
                mode="markers+text",
                marker=dict(size=18, color=color, symbol=symbol,
                            line=dict(color="white", width=1.5)),
                text=[f"R{rid}"],
                textposition="top center",
                textfont=dict(color="white", size=9),
                name=f"Robot {rid} ({s['status'].lower()})",
                showlegend=True,
            ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,26,46,1)",
        height=450,
        xaxis=dict(range=[-0.5, W - 0.5], showgrid=False, zeroline=False,
                   showticklabels=False, title=""),
        yaxis=dict(range=[H - 0.5, -0.5], showgrid=False, zeroline=False,
                   showticklabels=False, title="", scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=-0.05, font=dict(size=9)),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Robot status table ────────────────────────────────────────────────────────
st.subheader("Robot Status at Selected Step")
if step < len(history):
    snap_df = pd.DataFrame(history[step])
    st.dataframe(snap_df, use_container_width=True, hide_index=True)

st.divider()

# ── Congestion Heatmap ────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🔥 Congestion Heatmap")
    cong_pivot = congestion_df.pivot(index="row", columns="col", values="visits").fillna(0)
    fig_cong = px.imshow(cong_pivot, color_continuous_scale="Hot",
                         aspect="equal", template="plotly_dark", height=300)
    fig_cong.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Column", yaxis_title="Row",
        coloraxis_colorbar_title="Visits",
    )
    st.plotly_chart(fig_cong, use_container_width=True)

with col_b:
    st.subheader("📊 Tasks Completed Over Time")
    # Build cumulative completions from paths_df
    if len(paths_df) > 0:
        completed_by_step = (
            paths_df.groupby(["step", "robot_id"])["tasks_completed"]
            .max()
            .reset_index()
        )
        total_by_step = completed_by_step.groupby("step")["tasks_completed"].sum().reset_index()
        fig_tasks = px.line(total_by_step, x="step", y="tasks_completed",
                            template="plotly_dark", height=300,
                            color_discrete_sequence=["#44bb44"])
        fig_tasks.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,45,0.8)",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Time Step", yaxis_title="Cumulative Tasks Completed",
        )
        st.plotly_chart(fig_tasks, use_container_width=True)
    else:
        st.info("No task completion data available")
