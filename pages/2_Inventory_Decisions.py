"""Inventory intelligence page."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
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
except ImportError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
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
    reorder_only = st.toggle("Show only products that need reordering", value=False)
    selected_sku = st.selectbox("Product to explain", inventory_df["sku"].tolist())

render_header(
    "Inventory Policy",
    "Inventory Intelligence",
    "This page shows which products need reordering, why they were flagged, and how current stock compares with expected demand.",
)

filtered = inventory_df[inventory_df["stockout_risk"].isin(risk_filter)].copy()
if reorder_only:
    filtered = filtered[filtered["reorder_needed"] == 1]

focus = inventory_df[inventory_df["sku"] == selected_sku].iloc[0]

st.markdown(
    f"""
    <div class="panel panel-hero" style="margin-bottom:1.35rem;">
        <div class="page-kicker" style="margin-bottom:0.35rem;">Stock Summary</div>
        <div style="display:flex; justify-content:space-between; gap:1.5rem; align-items:end;">
            <div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.95rem; font-weight:700; color:#f4f6ff; line-height:1.06;">
                    {int(inventory_df['reorder_needed'].sum())} products need reordering and {int((inventory_df['stockout_risk'] == 'critical').sum())} are already in the highest risk band.
                </div>
                <div class="page-subtitle" style="margin-top:0.65rem; margin-bottom:0;">
                    {focus['sku']} is the selected example. It has {focus['coverage_days']:.1f} days of stock cover left, versus a reorder point of {focus['reorder_point']:.0f} units and a target stock level of {focus['target_stock']:.0f}.
                </div>
            </div>
            <div style="min-width:250px;">
                <div class="security-pill">{focus['stockout_risk'].upper()} risk</div>
                <div class="page-meta-bottom" style="margin-top:0.65rem;">Lead time {int(focus['lead_time_days'])}D // Service target {focus['service_level']:.1%}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander(f"How shortage risk is calculated · {focus['sku']}"):
    st.markdown(
        f"Shortage risk is based on **current stock**, **forecast demand**, **lead time**, and the **service level target**. "
        f"For **{focus['sku']}**, current stock of **{focus['current_stock']:.0f}** units is compared against expected demand "
        f"during a **{int(focus['lead_time_days'])}-day** replenishment window. "
        f"Reorder point: **{focus['reorder_point']:.0f}** units · Suggested order: **{focus['reorder_qty']:.0f}** units · "
        f"Estimated shortage: **{focus['shortage_units']:.0f}** units."
    )

for col, html in zip(
    st.columns(4),
    [
        metric_panel("Products To Reorder", f"{int(inventory_df['reorder_needed'].sum())}", "Products already below their reorder level"),
        metric_panel("Products In Immediate Danger", f"{int((inventory_df['stockout_risk'] == 'critical').sum())}", "Items most likely to stock out first"),
        metric_panel("Typical Stock Cover", f"{inventory_df['coverage_days'].median():.1f} days", "Typical number of days current stock will last"),
        metric_panel("Average Safety Buffer", f"{inventory_df['safety_stock'].mean():.0f}", "Extra units held to protect service levels"),
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
    display = display.rename(
        columns={
            "sku": "Product",
            "current_stock": "Current Stock",
            "reorder_point": "Reorder Level",
            "target_stock": "Target Stock",
            "reorder_qty": "Suggested Order",
            "coverage_days": "Days Of Stock Cover",
            "days_to_stockout": "Days Until Stockout",
            "stockout_risk": "Risk Level",
        }
    )
    display["Risk Level"] = display["Risk Level"].str.title()
    st.markdown('<div class="panel panel-tight">', unsafe_allow_html=True)
    st.subheader("Products Flagged For Review")
    st.markdown("<div class='table-note'>This table compares stock on hand with reorder level, target stock, and expected time to stockout</div>", unsafe_allow_html=True)
    dense_dataframe(display, height=420)
    st.markdown("</div>", unsafe_allow_html=True)

    risky = inventory_df.sort_values(["reorder_needed", "days_to_stockout"], ascending=[False, True]).head(10)
    compare = go.Figure()
    compare.add_trace(go.Bar(x=risky["sku"], y=risky["current_stock"], name="Current stock", marker_color="#22c55e"))
    compare.add_trace(go.Bar(x=risky["sku"], y=risky["reorder_point"], name="Reorder point", marker_color="#f97316"))
    compare.add_trace(go.Bar(x=risky["sku"], y=risky["target_stock"], name="Target stock", marker_color="#64748b"))
    compare = style_plotly(compare, 350)
    compare.update_layout(showlegend=False, barmode="group", xaxis_title="", yaxis_title="Units in stock")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Current Stock vs Reorder Level")
    st.caption("If the green bar is below the reorder point, the product is already in the reorder queue.")
    st.markdown(
        compact_legend(
            [
                ("Current stock", "#22c55e"),
                ("Reorder point", "#f97316"),
                ("Target stock", "#64748b"),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.plotly_chart(compare, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Why The Selected Product Was Flagged")
    st.markdown(f"<div class='note' style='margin-bottom:0.9rem;'><span class='tag'>{focus['sku']}</span>{focus['explanation']}</div>", unsafe_allow_html=True)
    st.markdown(
        status_rail(
            [
                ("Supplier lead time", f"{int(focus['lead_time_days'])} days"),
                ("Service target", f"{focus['service_level']:.1%}"),
                ("Expected sales, next 7 days", f"{focus['forecast_7d_total']:.0f}"),
                ("Estimated shortage", f"{focus['shortage_units']:.0f}"),
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
        color_discrete_sequence=["#3b82f6", "#22c55e", "#f97316", "#a78bfa", "#f43f5e"],
    )
    line = style_plotly(line, 355)
    line.update_layout(showlegend=False, yaxis_title="Expected stock on hand", xaxis_title="")
    line.add_hline(y=0, line_color="#ef4444", line_dash="dot", opacity=0.7, line_width=1)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Projected Stock Run-Down")
    st.caption("These lines show how quickly stock is expected to fall if demand continues as forecast. The red line marks the stockout point.")
    st.markdown(
        compact_legend(
            [(sku, color) for sku, color in zip(top_timeline["sku"].drop_duplicates().tolist(), ["#3b82f6", "#22c55e", "#f97316", "#a78bfa", "#f43f5e"])]
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
        color_discrete_map={"critical": "#ef4444", "high": "#f97316", "medium": "#eab308", "low": "#22c55e"},
    )
    risk_fig = style_plotly(risk_fig, 270)
    risk_fig.update_xaxes(categoryorder="array", categoryarray=["critical", "high", "medium", "low"])
    risk_fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Number of products")
    risk_fig.update_traces(marker_line_color="rgba(255,255,255,0.3)", marker_line_width=0.8)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Number Of Products In Each Risk Band")
    st.plotly_chart(risk_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
