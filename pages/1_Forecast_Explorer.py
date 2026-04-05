"""Forecast explorer page."""

from __future__ import annotations

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

apply_page_config("Forecast Explorer")
data = get_pipeline_data()
render_sidebar("forecast", data)

raw_df = data["raw_df"]
test_df = data["test_df"]
future_df = data["future_df"]
reliability_df = data["reliability_df"]

with st.sidebar:
    st.divider()
    selected_sku = st.selectbox("SKU", sorted(raw_df["sku"].unique()))
    horizon = st.segmented_control("Horizon", options=[7, 30], default=30)
    history_days = st.slider("History window", 90, 365, 180, step=15)
    show_bands = st.toggle("Show prediction bands", value=True)

render_header(
    "Recursive Forecasting",
    f"Forecast Explorer · {selected_sku}",
    "A SKU-level forecasting workbench with recursive horizon logic, confidence diagnostics, and operational context for price and promotion assumptions.",
)

sku_hist = raw_df[raw_df["sku"] == selected_sku].sort_values("date").tail(history_days)
sku_test = test_df[test_df["sku"] == selected_sku].sort_values("date")
sku_future = future_df[(future_df["sku"] == selected_sku) & (future_df["horizon_day"] <= horizon)].sort_values("date")
reliability = reliability_df[reliability_df["sku"] == selected_sku]
reliability_row = reliability.iloc[0] if not reliability.empty else None

st.markdown(
    f"""
    <div class="panel panel-hero" style="margin-bottom:1.35rem;">
        <div class="page-kicker" style="margin-bottom:0.35rem;">Forecast Intelligence Brief</div>
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
                <div class="security-pill">{reliability_row['reliability_category'] if reliability_row is not None else 'No'} reliability</div>
                <div class="page-meta-bottom" style="margin-top:0.65rem;">Promo days {int(sku_future['promotion'].sum())} // Avg price {sku_future['price'].mean():.2f}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

for col, html in zip(
    st.columns(4),
    [
        metric_panel("Observed Mean", f"{sku_hist['demand'].mean():.1f}", "Average daily realized demand"),
        metric_panel(f"{horizon}-Day Forecast", f"{sku_future['forecast'].sum():,.0f}", "Recursive horizon aggregate"),
        metric_panel("Reliability", f"{reliability_row['reliability_score']:.0%}" if reliability_row is not None else "N/A", reliability_row["reliability_category"] if reliability_row is not None else "No score"),
        metric_panel("Price Assumption", f"{sku_future['price'].mean():.2f}", "Mean forward selling price"),
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
        line=dict(color="#a39afc", width=2.1),
    )
)
if not sku_test.empty:
    chart.add_trace(
        go.Scatter(
            x=sku_test["date"],
            y=sku_test["forecast"],
            mode="lines",
            name="Holdout forecast",
            line=dict(color="#8bd7bd", width=1.9, dash="dot"),
        )
    )
if not sku_future.empty:
    chart.add_trace(
        go.Scatter(
            x=sku_future["date"],
            y=sku_future["forecast"],
            mode="lines+markers",
            name="Forward forecast",
            line=dict(color="#f4f0ff", width=2.4),
            marker=dict(size=5, color="#c7c2ff"),
        )
    )
if show_bands and not sku_future.empty:
    chart.add_trace(
        go.Scatter(
            x=list(sku_future["date"]) + list(sku_future["date"])[::-1],
            y=list(sku_future["upper"]) + list(sku_future["lower"])[::-1],
            fill="toself",
            fillcolor="rgba(143,132,255,0.14)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Prediction interval",
        )
    )
chart.add_vline(x=data["train_cutoff"], line_color="#ffc57f", line_dash="dash")
chart = style_plotly(chart, 470)
chart.update_layout(showlegend=False, yaxis_title="Units", xaxis_title="")

left, right = st.columns([1.65, 0.95], gap="large")
with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Demand and Forecast Path")
    st.markdown(
        compact_legend(
            [
                ("History", "#a39afc"),
                ("Holdout forecast", "#8bd7bd"),
                ("Forward forecast", "#f4f0ff"),
                ("Interval", "#8f84ff"),
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
    st.markdown('<div class="panel panel-tight">', unsafe_allow_html=True)
    st.subheader("Forward Schedule")
    st.markdown("<div class='table-note'>Dense planning ledger for the active forecast horizon</div>", unsafe_allow_html=True)
    dense_dataframe(table, height=330)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Reliability Reading")
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
                    ("Historical WAPE", f"{reliability_row['historical_wape']:.1%}"),
                    ("Coverage", f"{reliability_row['coverage']:.1%}"),
                    ("Width Ratio", f"{reliability_row['interval_width_ratio']:.1%}"),
                    ("Stability", f"{1 - reliability_row['recent_stability']:.1%}"),
                ]
            ),
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    recent = sku_test.tail(21).copy()
    if not recent.empty:
        recent["abs_error"] = (recent["demand"] - recent["forecast"]).abs()
        error_chart = go.Figure()
        error_chart.add_trace(go.Bar(x=recent["date"], y=recent["abs_error"], marker_color="#ff8d9f", name="Absolute error"))
        error_chart = style_plotly(error_chart, 285)
        error_chart.update_layout(showlegend=False, yaxis_title="Units", xaxis_title="")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Recent Holdout Error")
        st.plotly_chart(error_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    assumptions = sku_future[["promotion", "price"]].copy()
    assumptions["step"] = range(1, len(assumptions) + 1)
    assumption_chart = go.Figure()
    assumption_chart.add_trace(go.Scatter(x=assumptions["step"], y=assumptions["price"], mode="lines", name="Price", line=dict(color="#ffc57f", width=2)))
    assumption_chart.add_trace(go.Bar(x=assumptions["step"], y=assumptions["promotion"], name="Promotion", marker_color="rgba(143,132,255,0.48)", yaxis="y2"))
    assumption_chart = style_plotly(assumption_chart, 285)
    assumption_chart.update_layout(
        showlegend=False,
        xaxis_title="Forecast step",
        yaxis_title="Price",
        yaxis2=dict(overlaying="y", side="right", range=[0, 1.2], showgrid=False, title="Promo"),
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Forward Assumptions")
    st.plotly_chart(assumption_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
