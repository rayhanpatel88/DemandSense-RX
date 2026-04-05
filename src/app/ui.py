"""Shared Streamlit UI helpers for DemandSense-RX."""

from __future__ import annotations

import streamlit as st

from src.pipeline import run_pipeline
from src.utils.config import load_config

PAGE_TITLES = {
    "overview": "Executive Overview",
    "forecast": "Forecast Explorer",
    "inventory": "Inventory Intelligence",
    "robotics": "Warehouse Robotics",
    "explainability": "Explainability",
    "backtesting": "Backtesting",
    "scenario": "Scenario Simulator",
}

PLOTLY_THEME = {
    "paper_bgcolor": "#f7f5f1",
    "plot_bgcolor": "#f7f5f1",
    "font": {"family": "Georgia, Times New Roman, serif", "color": "#161616"},
}


@st.cache_resource(show_spinner="Running DemandSense-RX pipeline...")
def get_pipeline_data() -> dict:
    return run_pipeline(load_config())


def apply_page_config(page_title: str) -> None:
    st.set_page_config(page_title=page_title, page_icon="■", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@500;600&display=swap');
        :root{
            --bg:#f3f1ec;
            --panel:#faf8f4;
            --line:#dad4ca;
            --text:#141414;
            --muted:#6d685f;
            --accent:#1f3a5f;
            --accent-soft:#dce5ef;
            --alert:#a64032;
            --warn:#ad7b2b;
            --ok:#43624f;
        }
        .stApp { background: radial-gradient(circle at top left, #fcfbf8 0%, var(--bg) 55%, #ece7de 100%); color: var(--text); }
        [data-testid="stSidebar"] { background: #161616; border-right: 1px solid #2e2e2e; }
        [data-testid="stSidebar"] * { color: #f6f3ed !important; font-family: 'IBM Plex Sans', sans-serif !important; }
        html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif; }
        h1, h2, h3 { font-family: 'IBM Plex Serif', serif; letter-spacing: -0.02em; color: var(--text); }
        .page-shell { padding-top: 0.8rem; }
        .page-kicker { text-transform: uppercase; letter-spacing: 0.16em; color: var(--muted); font-size: 0.74rem; font-weight: 600; }
        .page-title { font-family: 'IBM Plex Serif', serif; font-size: 2.6rem; line-height: 1.05; margin: 0.2rem 0 0.35rem 0; }
        .page-subtitle { color: var(--muted); font-size: 1rem; max-width: 60rem; margin-bottom: 1.2rem; }
        .panel { background: rgba(250,248,244,0.94); border: 1px solid var(--line); border-radius: 22px; padding: 1.05rem 1.1rem; box-shadow: 0 12px 40px rgba(22,22,22,0.05); }
        .metric-panel { background: rgba(250,248,244,0.92); border: 1px solid var(--line); border-radius: 20px; padding: 1rem 1.1rem; min-height: 120px; box-shadow: 0 10px 26px rgba(22,22,22,0.05); }
        .metric-label { text-transform: uppercase; letter-spacing: 0.14em; color: var(--muted); font-size: 0.68rem; font-weight: 700; }
        .metric-value { font-family: 'IBM Plex Serif', serif; font-size: 2rem; line-height: 1.1; margin-top: 0.45rem; color: var(--text); }
        .metric-detail { margin-top: 0.55rem; color: var(--muted); font-size: 0.9rem; }
        .note { color: var(--muted); font-size: 0.9rem; line-height: 1.5; }
        .tag { display:inline-block; padding:0.25rem 0.55rem; border-radius:999px; border:1px solid var(--line); font-size:0.76rem; color:var(--muted); margin-right:0.35rem; }
        div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 18px; overflow: hidden; }
        div[data-testid="stMetric"] { background: transparent; border: none; }
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(active_key: str, data: dict | None = None) -> None:
    with st.sidebar:
        st.markdown("## DemandSense-RX")
        if data is not None:
            future_df = data["future_df"]
            inventory_df = data["inventory_df"]
            st.caption(
                f"{int(future_df['forecast'].sum()):,} forecast units | "
                f"{int(inventory_df['reorder_needed'].sum())} SKUs need action"
            )
        st.divider()
        st.page_link("streamlit_app.py", label=PAGE_TITLES["overview"], disabled=active_key == "overview")
        st.page_link("pages/1_Forecast_Explorer.py", label=PAGE_TITLES["forecast"], disabled=active_key == "forecast")
        st.page_link("pages/2_Inventory_Decisions.py", label=PAGE_TITLES["inventory"], disabled=active_key == "inventory")
        st.page_link("pages/3_Robotics_Simulation.py", label=PAGE_TITLES["robotics"], disabled=active_key == "robotics")
        st.page_link("pages/4_Explainability.py", label=PAGE_TITLES["explainability"], disabled=active_key == "explainability")
        st.page_link("pages/5_Backtesting.py", label=PAGE_TITLES["backtesting"], disabled=active_key == "backtesting")
        st.page_link("pages/6_Scenario_Simulator.py", label=PAGE_TITLES["scenario"], disabled=active_key == "scenario")


def render_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown('<div class="page-shell">', unsafe_allow_html=True)
    st.markdown(f'<div class="page-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def metric_panel(label: str, value: str, detail: str) -> str:
    return (
        '<div class="metric-panel">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-detail">{detail}</div>'
        "</div>"
    )


def style_plotly(fig, height: int = 320):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), **PLOTLY_THEME)
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#c6c0b5")
    fig.update_yaxes(showgrid=True, gridcolor="#e4dfd6", zeroline=False, linecolor="#c6c0b5")
    return fig
