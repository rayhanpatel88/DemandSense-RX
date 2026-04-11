"""Forecast explorer page."""

from __future__ import annotations

import sys
from pathlib import Path

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
        segmented_control,
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
        segmented_control,
        status_rail,
        style_plotly,
    )

apply_page_config("Forecast Explorer")
data = get_pipeline_data()
render_sidebar("forecast", data)

raw_df = data["raw_df"]
test_df = data["test_df"]
future_df = data["future_df"]
reliability_df = data["reliability_df"]

with st.sidebar:
    st.divider()
    selected_sku = st.selectbox("Product", sorted(raw_df["sku"].unique()))
    horizon = segmented_control("Planning window", options=[7, 30], default=30)
    history_days = st.slider("History shown", 90, 365, 180, step=15)
    show_bands = st.toggle("Show forecast range", value=True)

render_header(
    "Recursive Forecasting",
    f"Forecast Explorer · {selected_sku}",
    "This page shows how the forecast for one product is built, how reliable it is, and which assumptions are influencing it.",
)

sku_hist = raw_df[raw_df["sku"] == selected_sku].sort_values("date").tail(history_days)
sku_test = test_df[test_df["sku"] == selected_sku].sort_values("date")
sku_future = future_df[(future_df["sku"] == selected_sku) & (future_df["horizon_day"] <= horizon)].sort_values("date")
reliability = reliability_df[reliability_df["sku"] == selected_sku]
reliability_row = reliability.iloc[0] if not reliability.empty else None

st.markdown(
    f"""
    <div class="panel panel-hero" style="margin-bottom:1.35rem;">
        <div class="page-kicker" style="margin-bottom:0.35rem;">Forecast Summary</div>
        <div style="display:flex; justify-content:space-between; gap:1.5rem; align-items:end;">
            <div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.95rem; font-weight:700; color:#f4f6ff; line-height:1.06;">
                    {selected_sku} is projected for {sku_future['forecast'].sum():,.0f} units over the next {horizon} days.
                </div>
                <div class="page-subtitle" style="margin-top:0.65rem; margin-bottom:0;">
                    Recent realized demand averages {sku_hist['demand'].mean():.1f} units per day. The forward path updates lag and rolling state recursively rather than freezing features at the last historical row.
                </div>
            </div>
            <div style="min-width:250px;">
                <div class="security-pill">{reliability_row['reliability_category'] if reliability_row is not None else 'No'} forecast confidence</div>
                <div class="page-meta-bottom" style="margin-top:0.65rem;">Promotion days {int(sku_future['promotion'].sum())} // Average price {sku_future['price'].mean():.2f}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

for col, html in zip(
    st.columns(4),
    [
        metric_panel("Average Recent Daily Sales", f"{sku_hist['demand'].mean():.1f}", "Average units sold per day in recent history"),
        metric_panel(f"Expected Sales, Next {horizon} Days", f"{sku_future['forecast'].sum():,.0f}", "Total forecast for the selected planning window"),
        metric_panel("Forecast Confidence", f"{reliability_row['reliability_score']:.0%}" if reliability_row is not None else "N/A", reliability_row["reliability_category"] if reliability_row is not None else "No score"),
        metric_panel("Average Planned Price", f"{sku_future['price'].mean():.2f}", "Average selling price assumed in the forecast"),
    ],
):
    with col:
        st.markdown(html, unsafe_allow_html=True)

chart = go.Figure()
chart.add_trace(
    go.Scatter(
        x=sku_hist["date"],
        y=sku_hist["demand"],
        mode="lines",
        name="History",
        line=dict(color="#4b6480", width=1.8),
    )
)
if not sku_test.empty:
    chart.add_trace(
        go.Scatter(
            x=sku_test["date"],
            y=sku_test["forecast"],
            mode="lines",
            name="Holdout forecast",
            line=dict(color="#22c55e", width=1.8, dash="dot"),
        )
    )
if not sku_future.empty:
    chart.add_trace(
        go.Scatter(
            x=sku_future["date"],
            y=sku_future["forecast"],
            mode="lines+markers",
            name="Forward forecast",
            line=dict(color="#3b82f6", width=2.2),
            marker=dict(size=4, color="#60a5fa"),
        )
    )
if show_bands and not sku_future.empty:
    chart.add_trace(
        go.Scatter(
            x=list(sku_future["date"]) + list(sku_future["date"])[::-1],
            y=list(sku_future["upper"]) + list(sku_future["lower"])[::-1],
            fill="toself",
            fillcolor="rgba(59,130,246,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Prediction interval",
        )
    )
chart.add_vline(x=data["train_cutoff"], line_color="#f97316", line_dash="dash", line_width=1)
chart = style_plotly(chart, 470)
chart.update_layout(showlegend=False, yaxis_title="Units sold", xaxis_title="")

left, right = st.columns([1.65, 0.95], gap="large")
with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Recent Sales And Expected Sales")
    st.markdown(
        compact_legend(
            [
                ("History", "#4b6480"),
                ("Holdout forecast", "#22c55e"),
                ("Forward forecast", "#3b82f6"),
                ("Interval", "rgba(59,130,246,0.3)"),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.plotly_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    table = sku_future[["date", "forecast", "lower", "upper", "promotion", "price"]].copy()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    table["forecast"] = table["forecast"].round(1)
    table["lower"] = table["lower"].round(1)
    table["upper"] = table["upper"].round(1)
    table["price"] = table["price"].round(2)
    table = table.rename(
        columns={
            "date": "Date",
            "forecast": "Expected Sales",
            "lower": "Low Case",
            "upper": "High Case",
            "promotion": "Promotion Running",
            "price": "Planned Price",
        }
    )
    st.markdown('<div class="panel panel-tight">', unsafe_allow_html=True)
    st.subheader("Day-By-Day Forecast")
    st.markdown("<div class='table-note'>Daily forecast, expected range, promotion days, and planned price</div>", unsafe_allow_html=True)
    dense_dataframe(table, height=330)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("How Much To Trust This Forecast")
    if reliability_row is None:
        st.markdown("<div class='note'>No reliability summary was generated for this SKU.</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='note' style='margin-bottom:0.85rem;'><span class='tag'>{reliability_row['reliability_category']}</span>{reliability_row['reliability_explanation']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            status_rail(
                [
                    ("Average forecast miss", f"{reliability_row['historical_wape']:.1%}"),
                    ("Forecast range coverage", f"{reliability_row['coverage']:.1%}"),
                    ("Forecast range width", f"{reliability_row['interval_width_ratio']:.1%}"),
                    ("Recent consistency", f"{1 - reliability_row['recent_stability']:.1%}"),
                ]
            ),
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    recent = sku_test.tail(21).copy()
    if not recent.empty:
        recent["abs_error"] = (recent["demand"] - recent["forecast"]).abs()
        error_chart = go.Figure()
        error_chart.add_trace(go.Bar(x=recent["date"], y=recent["abs_error"], marker_color="#ef4444", name="Absolute error"))
        error_chart = style_plotly(error_chart, 285)
        error_chart.update_layout(showlegend=False, yaxis_title="Units missed", xaxis_title="")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Recent Forecast Misses")
        st.plotly_chart(error_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    assumptions = sku_future[["promotion", "price"]].copy()
    assumptions["step"] = range(1, len(assumptions) + 1)
    assumption_chart = go.Figure()
    assumption_chart.add_trace(go.Scatter(x=assumptions["step"], y=assumptions["price"], mode="lines", name="Price", line=dict(color="#f97316", width=1.8)))
    assumption_chart.add_trace(go.Bar(x=assumptions["step"], y=assumptions["promotion"], name="Promotion", marker_color="rgba(59,130,246,0.35)", yaxis="y2"))
    assumption_chart = style_plotly(assumption_chart, 285)
    assumption_chart.update_layout(
        showlegend=False,
        xaxis_title="Day in forecast",
        yaxis_title="Price",
        yaxis2=dict(overlaying="y", side="right", range=[0, 1.2], showgrid=False, title="Promotion"),
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Price And Promotion Assumptions")
    st.plotly_chart(assumption_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
