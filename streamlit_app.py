"""Nexar executive overview."""

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


apply_page_config("Nexar")
with st.spinner("Loading demand intelligence and decision layers..."):
    data = get_pipeline_data()

render_sidebar("overview", data)
render_header(
    "Executive Summary",
    "Business Planning Overview",
    "This page shows expected demand, stock risk, and the products that need attention first in plain business terms.",
)

future_df = data["future_df"]
raw_df = data["raw_df"]
inventory_df = data["inventory_df"]
reliability_df = data["reliability_df"]
slotting_df = data["slotting_df"]
future_7d = data["future_7d_df"]

forecast_total = int(future_df["forecast"].sum())
active_skus = int(raw_df["sku"].nunique())
reorder_count = int(inventory_df["reorder_needed"].sum())
high_risk = int(inventory_df["stockout_risk"].isin(["critical", "high"]).sum())
critical_risk = int(inventory_df["stockout_risk"].eq("critical").sum())
reliability_score = reliability_df["reliability_score"].mean() if not reliability_df.empty else 0.0
wape = float(data["holdout_metrics"].get("WAPE", 0.0))
avg_days_to_stockout = float(inventory_df["days_to_stockout"].replace([float("inf")], None).dropna().mean()) if "days_to_stockout" in inventory_df else 0.0

if critical_risk > 0:
    top_message = f"{critical_risk} products are at immediate risk of stocking out."
elif high_risk > 0:
    top_message = f"{high_risk} products need attention soon to avoid missed sales."
else:
    top_message = "No products are currently in the highest stock risk bands."

st.markdown(
    f"""
    <div class="panel panel-hero">
        <div class="page-kicker" style="margin-bottom:0.35rem;">What Needs Attention</div>
        <div style="display:grid; grid-template-columns:minmax(0, 1.7fr) minmax(250px, 0.95fr); gap:1.15rem; align-items:stretch;">
            <div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:clamp(1.7rem, 3.6vw, 2.7rem); font-weight:700; color:var(--text); line-height:1.08;">
                    {top_message}
                </div>
                <div class="page-subtitle" style="margin-top:0.8rem; margin-bottom:0;">
                    We expect demand of <strong>{forecast_total:,} units</strong> over the next 30 days across <strong>{active_skus}</strong> products.
                    Right now, <strong>{reorder_count}</strong> products are below their restocking threshold and should be reviewed first.
                </div>
                <div class="hero-badges" style="margin-top:1rem;">
                    <div class="hero-badge"><span class="hero-badge-dot"></span>{active_skus} active products</div>
                    <div class="hero-badge"><span class="hero-badge-dot"></span>{future_df['date'].nunique()} days ahead</div>
                    <div class="hero-badge"><span class="hero-badge-dot"></span>{slotting_df['zone'].nunique()} warehouse zones</div>
                </div>
            </div>
            <div class="glass-card" style="display:flex; flex-direction:column; justify-content:space-between; gap:0.9rem;">
                <div>
                    <div class="glass-label">Forecast Confidence</div>
                    <div class="glass-value">{reliability_score:.0%}</div>
                    <div class="page-meta-bottom" style="margin-top:0.35rem;">Average model confidence for the current planning run</div>
                </div>
                <div class="status-rail">
                    <div class="status-card">
                        <div class="status-card-label">Model Error</div>
                        <div class="status-card-value">{wape:.1f}%</div>
                    </div>
                    <div class="status-card">
                        <div class="status-card-label">Avg Days of Cover</div>
                        <div class="status-card-value">{avg_days_to_stockout:.0f}d</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

for col, html in zip(
    st.columns(4),
    [
        metric_panel("Expected Demand, Next 30 Days", f"{forecast_total:,}", "Total units we expect customers to buy"),
        metric_panel("Products To Reorder", f"{reorder_count}", "Items already below their restocking trigger"),
        metric_panel("Products At Risk", f"{high_risk}", "Items in high or critical stockout risk"),
        metric_panel("Forecast Confidence", f"{reliability_score:.0%}", f"Average reliability across products, with {wape:.1f}% model error"),
    ],
):
    with col:
        st.markdown(html, unsafe_allow_html=True)

with st.expander("What these numbers mean"):
    st.markdown(
        """
        **Demand** is the number of units we expect to sell.
        **Products to reorder** are already below their target stock level.
        **Products at risk** may run out before replenishment arrives.
        **Forecast confidence** tells you how much trust to place in the numbers.

        A product is flagged when current stock is not enough to cover expected demand during supplier lead time plus safety buffer.
        This is based on: current stock on hand, forecast demand, lead time in days, and the target service level.
        If stock falls below the reorder point, it enters the reorder queue. If projected stock hits zero before replenishment arrives, it moves into a higher shortage-risk band.
        """,
    )

main_left, main_right = st.columns([1.65, 1.0], gap="large")

with main_left:
    history = raw_df.groupby("date", as_index=False)["demand"].sum()
    forecast = future_df.groupby("date", as_index=False)["forecast"].sum()

    demand_fig = go.Figure()
    demand_fig.add_trace(
        go.Scatter(
            x=history.tail(180)["date"],
            y=history.tail(180)["demand"],
            mode="lines",
            name="Actual demand",
            line=dict(color="#4b6480", width=1.8),
        )
    )
    demand_fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["forecast"],
            mode="lines",
            name="Expected demand",
            line=dict(color="#3b82f6", width=2.4),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.1)",
        )
    )
    demand_fig = style_plotly(demand_fig, 390)
    demand_fig.update_layout(
        xaxis_title="",
        yaxis_title="Units",
        hovermode="x unified",
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Demand Trend And Next 30 Days")
    st.caption("The grey line shows what customers bought recently. The blue line shows what we expect them to buy next.")
    st.plotly_chart(demand_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    top_products = future_7d.groupby("sku", as_index=False)["forecast"].sum().sort_values("forecast", ascending=False).head(10)
    top_products["sku"] = top_products["sku"].astype(str)
    pressure_fig = px.bar(
        top_products.sort_values("forecast"),
        x="forecast",
        y="sku",
        orientation="h",
        text_auto=".0f",
        color_discrete_sequence=["#3b82f6"],
    )
    pressure_fig = style_plotly(pressure_fig, 355)
    pressure_fig.update_layout(
        showlegend=False,
        xaxis_title="Expected units over the next 7 days",
        yaxis_title="",
    )
    pressure_fig.update_traces(textposition="outside", marker_line_color="rgba(255,255,255,0.45)", marker_line_width=1.1)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Top Products Driving Demand This Week")
    st.caption("These products are expected to account for the most near-term sales, so they deserve the closest stock monitoring.")
    st.plotly_chart(pressure_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with main_right:
    risk_order = ["critical", "high", "medium", "low"]
    risk_mix = inventory_df["stockout_risk"].value_counts().reindex(risk_order, fill_value=0).reset_index()
    risk_mix.columns = ["risk", "count"]
    risk_mix["label"] = risk_mix["risk"].str.title()
    risk_fig = px.bar(
        risk_mix,
        x="count",
        y="label",
        orientation="h",
        text="count",
        color="risk",
        color_discrete_map={"critical": "#ef4444", "high": "#f97316", "medium": "#94a3b8", "low": "#22c55e"},
    )
    risk_fig = style_plotly(risk_fig, 300)
    risk_fig.update_layout(showlegend=False, xaxis_title="Number of products", yaxis_title="")
    risk_fig.update_traces(textposition="outside")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("How Much Stock Risk We Have")
    st.caption("This shows how many products fall into each risk level, from critical down to low.")
    st.plotly_chart(risk_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    weakest = reliability_df.sort_values("reliability_score").head(4) if not reliability_df.empty else None
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Where To Treat The Forecast Carefully")
    st.caption("These products have the least reliable forecast signals, so decisions around them need extra judgement.")
    if weakest is not None and not weakest.empty:
        for _, row in weakest.iterrows():
            st.markdown(
                f"<div class='note' style='margin-bottom:0.85rem;'><span class='tag'>{row['sku']}</span>{row['reliability_explanation']}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<div class='note'>All products have usable forecast confidence summaries.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    urgent = inventory_df.sort_values(["reorder_needed", "days_to_stockout"], ascending=[False, True]).head(5)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Recommended Actions Right Now")
    st.caption("This is the short list of products that deserve immediate operational attention.")
    for _, row in urgent.iterrows():
        st.markdown(
            f"<div class='note' style='margin-bottom:0.85rem;'><span class='tag'>{row['sku']}</span>{row['explanation']}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

ops_left, ops_mid, ops_right = st.columns([1.1, 1.15, 0.95], gap="large")

with ops_left:
    reorder_table = inventory_df.sort_values(["reorder_needed", "days_to_stockout"], ascending=[False, True]).head(10).copy()
    reorder_table = reorder_table.rename(
        columns={
            "sku": "Product",
            "current_stock": "Current Stock",
            "reorder_point": "Reorder Level",
            "reorder_qty": "Suggested Order",
            "stockout_risk": "Risk Level",
        }
    )
    reorder_table["Risk Level"] = reorder_table["Risk Level"].str.title()
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Products To Review First")
    st.caption("A simple restocking queue showing which products are most likely to need action soon.")
    st.dataframe(
        reorder_table[["Product", "Current Stock", "Reorder Level", "Suggested Order", "Risk Level"]],
        use_container_width=True,
        hide_index=True,
        height=330,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with ops_mid:
    cover_df = slotting_df.head(15).copy()
    bubble_size = cover_df["forecast_30d_total"].clip(lower=1)
    cover_fig = go.Figure()
    cover_fig.add_trace(
        go.Scatter(
            x=cover_df["forecast_30d_total"],
            y=cover_df["days_to_stockout"],
            mode="markers+text",
            text=cover_df["sku"],
            textposition="top center",
            marker=dict(
                size=bubble_size / bubble_size.max() * 24 + 8,
                color=cover_df["days_to_stockout"],
                colorscale=[[0, "#dc2626"], [0.45, "#f97316"], [1, "#2563eb"]],
                line=dict(color="#ffffff", width=1),
                showscale=False,
            ),
        )
    )
    cover_fig = style_plotly(cover_fig, 330)
    cover_fig.update_layout(xaxis_title="Expected demand over the next 30 days", yaxis_title="Days until stockout")
    cover_fig.add_hline(y=14, line_color="#f97316", line_dash="dot", opacity=0.8)
    cover_fig.add_vline(x=cover_df["forecast_30d_total"].median(), line_color="#64748b", line_dash="dot", opacity=0.45)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("High Demand Products With Thin Stock Cover")
    st.caption("Products near the bottom-right are the most exposed: strong expected demand and not much time before stock runs out. The orange guide marks roughly two weeks of cover.")
    st.plotly_chart(cover_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with ops_right:
    hottest_zone = (
        future_7d.merge(slotting_df[["sku", "zone"]], on="sku", how="left")
        .groupby("zone")["forecast"]
        .sum()
        .sort_values(ascending=False)
    )
    top_zone = hottest_zone.index[0] if not hottest_zone.empty else "N/A"
    zone_share = (hottest_zone.iloc[0] / hottest_zone.sum()) if not hottest_zone.empty else 0.0
    top_zone_units = int(hottest_zone.iloc[0]) if not hottest_zone.empty else 0
    st.markdown(
        f"""
        <div class="panel" style="margin-bottom:1rem;">
            <div class="page-kicker" style="margin-bottom:0.4rem;">Warehouse Focus</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; line-height:1;">{top_zone}</div>
            <div class="page-subtitle" style="margin-top:0.55rem; margin-bottom:0.95rem;">
                This zone is expected to carry the most near-term demand and may need the cleanest replenishment flow.
            </div>
            <div class="status-rail">
                <div class="status-card">
                    <div class="status-card-label">Share Of 7-Day Demand</div>
                    <div class="status-card-value">{zone_share:.0%}</div>
                </div>
                <div class="status-card">
                    <div class="status-card-label">Expected Units</div>
                    <div class="status-card-value">{top_zone_units:,}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    snapshot_items = [
        ("Forecast confidence", f"{reliability_score:.0%}"),
        ("Model error", f"{wape:.1f}%"),
        ("Products at risk", f"{high_risk}"),
        ("Planning window", f"{future_df['date'].nunique()} days"),
    ]
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Quick Snapshot")
    st.caption("A short plain-language summary of the current planning run.")
    st.markdown(
        "".join(
            [
                f"<div class='note' style='display:flex; justify-content:space-between; gap:0.75rem; padding:0.72rem 0; border-bottom:1px solid var(--line);'><span>{label}</span><strong style='color:var(--text);'>{value}</strong></div>"
                for label, value in snapshot_items
            ]
        ),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
