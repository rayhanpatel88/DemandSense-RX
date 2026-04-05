"""Inventory intelligence page."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app.ui import (
    apply_page_config,
    compact_legend,
    dense_dataframe,
    get_pipeline_data,
    metric_panel,
    render_header,
    render_sidebar,
    status_rail,
    style_plotly,
)

apply_page_config("Inventory Intelligence")
data = get_pipeline_data()
render_sidebar("inventory", data)

inventory_df = data["inventory_df"]
timeline_df = data["stockout_timeline"]

with st.sidebar:
    st.divider()
    risk_filter = st.multiselect("Risk levels", ["critical", "high", "medium", "low"], default=["critical", "high", "medium", "low"])
    reorder_only = st.toggle("Only reorder candidates", value=False)
    selected_sku = st.selectbox("Explanation focus", inventory_df["sku"].tolist())

render_header(
    "Inventory Policy",
    "Inventory Intelligence",
    "A replenishment decision surface built on service-level targets, demand variance, lead-time exposure, and forecast-linked stock trajectories.",
)

filtered = inventory_df[inventory_df["stockout_risk"].isin(risk_filter)].copy()
if reorder_only:
    filtered = filtered[filtered["reorder_needed"] == 1]

focus = inventory_df[inventory_df["sku"] == selected_sku].iloc[0]

st.markdown(
    f"""
    <div class="panel panel-hero" style="margin-bottom:1.35rem;">
        <div class="page-kicker" style="margin-bottom:0.35rem;">Strategic Stock Advisory</div>
        <div style="display:flex; justify-content:space-between; gap:1.5rem; align-items:end;">
            <div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.95rem; font-weight:700; color:#f4f6ff; line-height:1.06;">
                    {int(inventory_df['reorder_needed'].sum())} SKUs require action and {int((inventory_df['stockout_risk'] == 'critical').sum())} are already in critical exposure.
                </div>
                <div class="page-subtitle" style="margin-top:0.65rem; margin-bottom:0;">
                    {focus['sku']} is the current focus node. It carries {focus['coverage_days']:.1f} days of cover against a reorder point of {focus['reorder_point']:.0f} units and target stock of {focus['target_stock']:.0f}.
                </div>
            </div>
            <div style="min-width:250px;">
                <div class="security-pill">{focus['stockout_risk'].upper()} risk</div>
                <div class="page-meta-bottom" style="margin-top:0.65rem;">Lead time {int(focus['lead_time_days'])}D // Service {focus['service_level']:.1%}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

for col, html in zip(
    st.columns(4),
    [
        metric_panel("Reorder Candidates", f"{int(inventory_df['reorder_needed'].sum())}", "SKUs below reorder point"),
        metric_panel("Critical Exposure", f"{int((inventory_df['stockout_risk'] == 'critical').sum())}", "Immediate service risk"),
        metric_panel("Median Cover", f"{inventory_df['coverage_days'].median():.1f} days", "Current stock protection"),
        metric_panel("Mean Safety Stock", f"{inventory_df['safety_stock'].mean():.0f}", "Units held as variability buffer"),
    ],
):
    with col:
        st.markdown(html, unsafe_allow_html=True)

left, right = st.columns([1.6, 0.95], gap="large")
with left:
    display = filtered[
        [
            "sku",
            "current_stock",
            "reorder_point",
            "target_stock",
            "reorder_qty",
            "coverage_days",
            "days_to_stockout",
            "stockout_risk",
        ]
    ].copy()
    st.markdown('<div class="panel panel-tight">', unsafe_allow_html=True)
    st.subheader("Decision Ledger")
    st.markdown("<div class='table-note'>Filtered reorder queue with operating policy targets</div>", unsafe_allow_html=True)
    dense_dataframe(display, height=420)
    st.markdown("</div>", unsafe_allow_html=True)

    risky = inventory_df.sort_values(["reorder_needed", "days_to_stockout"], ascending=[False, True]).head(10)
    compare = go.Figure()
    compare.add_trace(go.Bar(x=risky["sku"], y=risky["current_stock"], name="Current stock", marker_color="#8bd7bd"))
    compare.add_trace(go.Bar(x=risky["sku"], y=risky["reorder_point"], name="Reorder point", marker_color="#ffc57f"))
    compare.add_trace(go.Bar(x=risky["sku"], y=risky["target_stock"], name="Target stock", marker_color="#a39afc"))
    compare = style_plotly(compare, 350)
    compare.update_layout(showlegend=False, barmode="group", xaxis_title="", yaxis_title="Units")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Policy Comparison")
    st.markdown(
        compact_legend(
            [
                ("Current stock", "#8bd7bd"),
                ("Reorder point", "#ffc57f"),
                ("Target stock", "#a39afc"),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.plotly_chart(compare, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Recommendation Narrative")
    st.markdown(f"<div class='note' style='margin-bottom:0.9rem;'><span class='tag'>{focus['sku']}</span>{focus['explanation']}</div>", unsafe_allow_html=True)
    st.markdown(
        status_rail(
            [
                ("Lead Time", f"{int(focus['lead_time_days'])} days"),
                ("Service Level", f"{focus['service_level']:.1%}"),
                ("7-Day Forecast", f"{focus['forecast_7d_total']:.0f}"),
                ("Shortage Units", f"{focus['shortage_units']:.0f}"),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    top_timeline = timeline_df[timeline_df["sku"].isin(inventory_df.sort_values("days_to_stockout").head(5)["sku"])].copy()
    line = px.line(
        top_timeline,
        x="date",
        y="projected_stock",
        color="sku",
        color_discrete_sequence=["#a39afc", "#8bd7bd", "#ffc57f", "#f4f0ff", "#ff8d9f"],
    )
    line = style_plotly(line, 355)
    line.update_layout(showlegend=False, yaxis_title="Projected stock", xaxis_title="")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Projected Depletion")
    st.markdown(
        compact_legend(
            [(sku, color) for sku, color in zip(top_timeline["sku"].drop_duplicates().tolist(), ["#a39afc", "#8bd7bd", "#ffc57f", "#f4f0ff", "#ff8d9f"])]
        ),
        unsafe_allow_html=True,
    )
    st.plotly_chart(line, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    risk_mix = inventory_df["stockout_risk"].value_counts().rename_axis("risk").reset_index(name="count")
    risk_fig = px.bar(
        risk_mix,
        x="risk",
        y="count",
        color="risk",
        color_discrete_map={"critical": "#ff8d9f", "high": "#ffc57f", "medium": "#a39afc", "low": "#8bd7bd"},
    )
    risk_fig = style_plotly(risk_fig, 270)
    risk_fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="SKU count")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Risk Mix")
    st.plotly_chart(risk_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
