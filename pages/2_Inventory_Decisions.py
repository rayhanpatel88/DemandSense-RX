"""Inventory intelligence page."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app.ui import apply_page_config, get_pipeline_data, metric_panel, render_header, render_sidebar, style_plotly

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
    "Service-level stock policy now uses demand variability, lead-time exposure, and forward demand coverage rather than a hand-wavy reorder heuristic.",
)

filtered = inventory_df[inventory_df["stockout_risk"].isin(risk_filter)].copy()
if reorder_only:
    filtered = filtered[filtered["reorder_needed"] == 1]

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

left, right = st.columns([1.55, 1.0], gap="large")
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
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Decision Table")
    st.dataframe(display, use_container_width=True, hide_index=True, height=420)
    st.markdown("</div>", unsafe_allow_html=True)

    risky = inventory_df.sort_values(["reorder_needed", "days_to_stockout"], ascending=[False, True]).head(10)
    compare = go.Figure()
    compare.add_trace(go.Bar(x=risky["sku"], y=risky["current_stock"], name="Current stock", marker_color="#1f3a5f"))
    compare.add_trace(go.Bar(x=risky["sku"], y=risky["reorder_point"], name="Reorder point", marker_color="#b4a06a"))
    compare.add_trace(go.Bar(x=risky["sku"], y=risky["target_stock"], name="Target stock", marker_color="#8c5a2b"))
    compare = style_plotly(compare, 340)
    compare.update_layout(barmode="group", legend=dict(orientation="h", y=1.08, x=0), xaxis_title="", yaxis_title="Units")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Policy Comparison")
    st.plotly_chart(compare, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    focus = inventory_df[inventory_df["sku"] == selected_sku].iloc[0]
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Recommendation Narrative")
    st.markdown(f"<div class='note'><span class='tag'>{focus['sku']}</span>{focus['explanation']}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='note'>Lead time {int(focus['lead_time_days'])} days | Service level {focus['service_level']:.1%} | "
        f"7-day forecast {focus['forecast_7d_total']:.0f} units</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    top_timeline = timeline_df[timeline_df["sku"].isin(inventory_df.sort_values("days_to_stockout").head(5)["sku"])].copy()
    line = px.line(top_timeline, x="date", y="projected_stock", color="sku", color_discrete_sequence=["#1f3a5f", "#8c5a2b", "#6d685f", "#5a715f", "#a64032"])
    line = style_plotly(line, 360)
    line.update_layout(legend=dict(orientation="h", y=1.08, x=0), yaxis_title="Projected stock", xaxis_title="")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Projected Depletion")
    st.plotly_chart(line, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
