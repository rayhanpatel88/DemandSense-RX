"""DemandSense-RX executive overview."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app.ui import apply_page_config, get_pipeline_data, metric_panel, render_header, render_sidebar, style_plotly

apply_page_config("DemandSense-RX")
data = get_pipeline_data()
render_sidebar("overview", data)
render_header(
    "Demand Sensing Platform",
    "Executive Overview",
    "A unified operating view across recursive demand forecasts, inventory posture, and warehouse execution pressure.",
)

future_df = data["future_df"]
raw_df = data["raw_df"]
inventory_df = data["inventory_df"]
reliability_df = data["reliability_df"]
backtest_summary = data["backtest_results"]["summary"]
slotting_df = data["slotting_df"]

forecast_total = int(future_df["forecast"].sum())
reorder_count = int(inventory_df["reorder_needed"].sum())
high_risk = int(inventory_df["stockout_risk"].isin(["critical", "high"]).sum())
reliability_score = reliability_df["reliability_score"].mean() if not reliability_df.empty else 0.0
wape = float(backtest_summary.loc["LightGBM", "WAPE"]) if "LightGBM" in backtest_summary.index else 0.0

for col, html in zip(
    st.columns(4),
    [
        metric_panel("30-Day Demand", f"{forecast_total:,}", f"{raw_df['sku'].nunique()} active SKUs in forecast scope"),
        metric_panel("Inventory Actions", f"{reorder_count}", "SKUs currently below reorder policy"),
        metric_panel("Execution Risk", f"{high_risk}", "High or critical stockout exposure"),
        metric_panel("Forecast Trust", f"{reliability_score:.0%}", f"Backtested recursive model WAPE {wape:.1f}%"),
    ],
):
    with col:
        st.markdown(html, unsafe_allow_html=True)

left, right = st.columns([1.6, 1.0], gap="large")

with left:
    history = raw_df.groupby("date")["demand"].sum().reset_index()
    history["series"] = "Historical"
    forecast = future_df.groupby("date")["forecast"].sum().reset_index()
    forecast["series"] = "Forecast"
    combined = history.tail(180).rename(columns={"demand": "units"})
    future_series = forecast.rename(columns={"forecast": "units"})
    timeline = px.line(
        combined.rename(columns={"date": "date", "units": "units", "series": "series"}),
        x="date",
        y="units",
        color="series",
        color_discrete_map={"Historical": "#1f3a5f"},
    )
    timeline.add_trace(
        go.Scatter(
            x=future_series["date"],
            y=future_series["units"],
            mode="lines",
            name="Forecast",
            line=dict(color="#8c5a2b", width=2.5),
        )
    )
    timeline = style_plotly(timeline, 380)
    timeline.update_layout(legend=dict(orientation="h", y=1.08, x=0), yaxis_title="Units", xaxis_title="")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Demand Trajectory")
    st.caption("Historical realized demand transitions into recursive forecast volume.")
    st.plotly_chart(timeline, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    top_flow = slotting_df.head(12).copy()
    top_flow["slot_label"] = top_flow["sku"] + "  ·  #" + top_flow["slot_rank"].astype(str)
    bar = px.bar(
        top_flow,
        x="forecast_30d_total",
        y="slot_label",
        orientation="h",
        color="forecast_30d_total",
        color_continuous_scale=["#dfe8f1", "#1f3a5f"],
    )
    bar = style_plotly(bar, 340)
    bar.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="30-day forecast units", yaxis={"categoryorder": "total ascending"})
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Warehouse Slotting Pressure")
    st.caption("High-volume SKUs rise to the top of the slotting plan and drive travel intensity.")
    st.plotly_chart(bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    risk_mix = inventory_df["stockout_risk"].value_counts().rename_axis("risk").reset_index(name="count")
    donut = px.pie(
        risk_mix,
        values="count",
        names="risk",
        hole=0.68,
        color="risk",
        color_discrete_map={"critical": "#a64032", "high": "#c07b38", "medium": "#b4a06a", "low": "#5a715f"},
    )
    donut = style_plotly(donut, 280)
    donut.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.12))
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Inventory Risk Mix")
    st.plotly_chart(donut, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Reliability Notes")
    weakest = reliability_df.sort_values("reliability_score").head(4) if not reliability_df.empty else None
    if weakest is not None and not weakest.empty:
        for _, row in weakest.iterrows():
            st.markdown(
                f"<div class='note'><span class='tag'>{row['sku']}</span>{row['reliability_explanation']}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<div class='note'>Reliability diagnostics were unavailable for this run.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Immediate Actions")
    urgent = inventory_df.sort_values(["reorder_needed", "days_to_stockout"], ascending=[False, True]).head(6)
    for _, row in urgent.iterrows():
        st.markdown(
            f"<div class='note'><span class='tag'>{row['sku']}</span>{row['explanation']}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
