"""Forecast explorer page."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.app.ui import apply_page_config, get_pipeline_data, metric_panel, render_header, render_sidebar, style_plotly

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
    "Each future step is generated recursively so the model’s own output updates lag and rolling-demand state before the next prediction.",
)

sku_hist = raw_df[raw_df["sku"] == selected_sku].sort_values("date").tail(history_days)
sku_test = test_df[test_df["sku"] == selected_sku].sort_values("date")
sku_future = future_df[(future_df["sku"] == selected_sku) & (future_df["horizon_day"] <= horizon)].sort_values("date")
reliability = reliability_df[reliability_df["sku"] == selected_sku]

for col, html in zip(
    st.columns(4),
    [
        metric_panel("Observed Mean", f"{sku_hist['demand'].mean():.1f}", "Average daily realized demand"),
        metric_panel(f"{horizon}-Day Forecast", f"{sku_future['forecast'].sum():,.0f}", "Recursive horizon aggregate"),
        metric_panel("Reliability", f"{reliability['reliability_score'].iloc[0]:.0%}" if not reliability.empty else "N/A", reliability["reliability_category"].iloc[0] if not reliability.empty else "No score"),
        metric_panel("Promo Days", f"{int(sku_future['promotion'].sum())}", "Forecast horizon promo assumptions"),
    ],
):
    with col:
        st.markdown(html, unsafe_allow_html=True)

chart = go.Figure()
chart.add_trace(go.Scatter(x=sku_hist["date"], y=sku_hist["demand"], mode="lines", name="History", line=dict(color="#1f3a5f", width=2)))
if not sku_test.empty:
    chart.add_trace(go.Scatter(x=sku_test["date"], y=sku_test["forecast"], mode="lines", name="Holdout forecast", line=dict(color="#6d685f", width=2, dash="dot")))
if not sku_future.empty:
    chart.add_trace(go.Scatter(x=sku_future["date"], y=sku_future["forecast"], mode="lines+markers", name="Forward forecast", line=dict(color="#8c5a2b", width=2.5)))
if show_bands and not sku_future.empty:
    chart.add_trace(
        go.Scatter(
            x=list(sku_future["date"]) + list(sku_future["date"])[::-1],
            y=list(sku_future["upper"]) + list(sku_future["lower"])[::-1],
            fill="toself",
            fillcolor="rgba(140,90,43,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Prediction interval",
        )
    )
chart.add_vline(x=data["train_cutoff"], line_color="#b4a06a", line_dash="dash")
chart = style_plotly(chart, 460)
chart.update_layout(legend=dict(orientation="h", y=1.08, x=0), yaxis_title="Units", xaxis_title="")

left, right = st.columns([1.75, 1.0], gap="large")
with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Demand and Forecast Path")
    st.plotly_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    table = sku_future[["date", "forecast", "lower", "upper", "promotion", "price"]].copy()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Forward Schedule")
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Reliability Reading")
    if reliability.empty:
        st.markdown("<div class='note'>No reliability summary was generated for this SKU.</div>", unsafe_allow_html=True)
    else:
        row = reliability.iloc[0]
        st.markdown(f"<div class='note'><span class='tag'>{row['reliability_category']}</span>{row['reliability_explanation']}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='note'>Historical WAPE {row['historical_wape']:.1%} | Interval coverage {row['coverage']:.1%} | "
            f"Width ratio {row['interval_width_ratio']:.1%}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    recent = sku_test.tail(21).copy()
    if not recent.empty:
        recent["abs_error"] = (recent["demand"] - recent["forecast"]).abs()
        error_chart = go.Figure()
        error_chart.add_trace(go.Bar(x=recent["date"], y=recent["abs_error"], marker_color="#a64032", name="Absolute error"))
        error_chart = style_plotly(error_chart, 280)
        error_chart.update_layout(showlegend=False, yaxis_title="Units", xaxis_title="")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Recent Holdout Error")
        st.plotly_chart(error_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
