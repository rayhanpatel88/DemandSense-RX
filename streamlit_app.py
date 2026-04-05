"""DemandSense-RX executive overview."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from src.app.ui import apply_page_config, get_pipeline_data, metric_panel, render_header, render_sidebar, style_plotly
except ImportError:
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.app.ui import apply_page_config, get_pipeline_data, metric_panel, render_header, render_sidebar, style_plotly

apply_page_config("DemandSense-RX")
data = get_pipeline_data()
render_sidebar("overview", data)
render_header(
    "Operational Command",
    "Executive Overview",
    "A portfolio-level control surface for demand sensing, stock exposure, and warehouse execution pressure. Every view is tied to the same recursive forecast and operating policy outputs.",
)

future_df = data["future_df"]
raw_df = data["raw_df"]
inventory_df = data["inventory_df"]
reliability_df = data["reliability_df"]
backtest_summary = data["backtest_results"]["summary"]
slotting_df = data["slotting_df"]
future_7d = data["future_7d_df"]

forecast_total = int(future_df["forecast"].sum())
reorder_count = int(inventory_df["reorder_needed"].sum())
high_risk = int(inventory_df["stockout_risk"].isin(["critical", "high"]).sum())
reliability_score = reliability_df["reliability_score"].mean() if not reliability_df.empty else 0.0
wape = float(backtest_summary.loc["LightGBM", "WAPE"]) if "LightGBM" in backtest_summary.index else 0.0

st.markdown(
    f"""
    <div class="panel panel-hero" style="margin-bottom:1.35rem;">
        <div class="page-kicker" style="margin-bottom:0.35rem;">Strategic Operating Advisory</div>
        <div style="display:flex; justify-content:space-between; gap:1.5rem; align-items:end;">
            <div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; color:#f4f6ff; line-height:1.05;">
                    Demand baseline remains elevated while inventory posture is still under-buffered.
                </div>
                <div class="page-subtitle" style="margin-top:0.65rem; margin-bottom:0;">
                    The forecast engine is signalling {forecast_total:,} units over the next 30 days. {reorder_count} SKUs require replenishment and {high_risk} remain in high or critical exposure bands.
                </div>
            </div>
            <div style="min-width:240px;">
                <div class="security-pill">Model Trust {reliability_score:.0%}</div>
                <div class="page-meta-bottom" style="margin-top:0.65rem;">Recursive WAPE {wape:.1f}% // Horizon 30D</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

for col, html in zip(
    st.columns(4),
    [
        metric_panel("30-Day Forecasted Demand", f"{forecast_total:,}", f"{raw_df['sku'].nunique()} active SKUs in planning scope"),
        metric_panel("Replenishment Queue", f"{reorder_count}", "Current stock below reorder policy threshold"),
        metric_panel("Operational Risk", f"{high_risk}", "High and critical stockout exposure"),
        metric_panel("Forecast Reliability", f"{reliability_score:.0%}", f"Backtested recursive model WAPE {wape:.1f}%"),
    ],
):
    with col:
        st.markdown(html, unsafe_allow_html=True)

main_left, main_right = st.columns([1.7, 0.95], gap="large")

with main_left:
    history = raw_df.groupby("date")["demand"].sum().reset_index()
    forecast = future_df.groupby("date")["forecast"].sum().reset_index()
    history_fig = go.Figure()
    history_fig.add_trace(
        go.Scatter(
            x=history.tail(180)["date"],
            y=history.tail(180)["demand"],
            mode="lines",
            name="Historical",
            line=dict(color="#a39afc", width=2.1),
        )
    )
    history_fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["forecast"],
            mode="lines",
            name="Recursive forecast",
            line=dict(color="#f1eefc", width=2.4),
            fill="tozeroy",
            fillcolor="rgba(163,154,252,0.14)",
        )
    )
    history_fig = style_plotly(history_fig, 390)
    history_fig.update_layout(legend=dict(orientation="h", y=1.08, x=0), xaxis_title="", yaxis_title="Units")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Demand Flux and Forecast Path")
    st.caption("Historical demand transitions directly into recursively generated forecast volume.")
    st.plotly_chart(history_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    zone_pressure = future_7d.groupby("sku")["forecast"].sum().sort_values(ascending=False).head(12).reset_index()
    zone_pressure["rank"] = zone_pressure.index + 1
    zone_pressure["label"] = zone_pressure["sku"] + " // #" + zone_pressure["rank"].astype(str)
    slot_fig = px.bar(
        zone_pressure,
        x="forecast",
        y="label",
        orientation="h",
        color="forecast",
        color_continuous_scale=["#233152", "#9b92ff"],
    )
    slot_fig = style_plotly(slot_fig, 340)
    slot_fig.update_layout(
        coloraxis_showscale=False,
        yaxis_title="",
        xaxis_title="Next 7-day demand intensity",
        yaxis={"categoryorder": "total ascending"},
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Short-Horizon Pressure Map")
    st.caption("These SKUs dominate near-term slotting and pick activity.")
    st.plotly_chart(slot_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with main_right:
    risk_mix = inventory_df["stockout_risk"].value_counts().rename_axis("risk").reset_index(name="count")
    risk_fig = px.bar(
        risk_mix,
        x="risk",
        y="count",
        color="risk",
        color_discrete_map={"critical": "#ff8d9f", "high": "#ffc57f", "medium": "#9aa7c8", "low": "#87d2bc"},
    )
    risk_fig = style_plotly(risk_fig, 280)
    risk_fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="SKU count")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Risk Ledger")
    st.plotly_chart(risk_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    weakest = reliability_df.sort_values("reliability_score").head(4) if not reliability_df.empty else None
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Reliability Exceptions")
    if weakest is not None and not weakest.empty:
        for _, row in weakest.iterrows():
            st.markdown(
                f"<div class='note' style='margin-bottom:0.8rem;'><span class='tag'>{row['sku']}</span>{row['reliability_explanation']}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<div class='note'>No reliability exceptions were generated.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    urgent = inventory_df.sort_values(["reorder_needed", "days_to_stockout"], ascending=[False, True]).head(6)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Action Queue")
    for _, row in urgent.iterrows():
        st.markdown(
            f"<div class='note' style='margin-bottom:0.8rem;'><span class='tag'>{row['sku']}</span>{row['explanation']}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

bottom_left, bottom_right = st.columns([1.2, 1.4], gap="large")
with bottom_left:
    top_inventory = inventory_df.head(10).copy()
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Reorder Queue")
    st.dataframe(
        top_inventory[["sku", "current_stock", "reorder_point", "reorder_qty", "stockout_risk"]],
        use_container_width=True,
        hide_index=True,
        height=330,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with bottom_right:
    slotting = slotting_df.head(15).copy()
    slotting["status"] = slotting["stockout_risk"].str.upper()
    status_fig = go.Figure()
    status_fig.add_trace(
        go.Scatter(
            x=slotting["forecast_30d_total"],
            y=slotting["days_to_stockout"],
            mode="markers+text",
            text=slotting["sku"],
            textposition="top center",
            marker=dict(
                size=slotting["forecast_30d_total"] / slotting["forecast_30d_total"].max() * 24 + 8,
                color=slotting["days_to_stockout"],
                colorscale=[[0, "#ff8d9f"], [0.45, "#ffc57f"], [1, "#8f84ff"]],
                line=dict(color="#0c1220", width=1.5),
                showscale=False,
            ),
        )
    )
    status_fig = style_plotly(status_fig, 330)
    status_fig.update_layout(xaxis_title="30-day forecast units", yaxis_title="Days to stockout")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Volume vs Cover")
    st.caption("High-volume, low-cover SKUs are the operational fault line.")
    st.plotly_chart(status_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
