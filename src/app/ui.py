"""Shared Streamlit UI helpers for Nexar."""

from __future__ import annotations

import base64
from pathlib import Path
from string import Template
from typing import Any, Optional

import streamlit as st

PAGE_TITLES = {
    "overview": "Executive Overview",
    "forecast": "Forecast Explorer",
    "inventory": "Inventory Intelligence",
    "robotics": "Warehouse Robotics",
    "explainability": "Explainability",
    "backtesting": "Backtesting",
    "scenario": "Scenario Simulator",
}

PAGE_ICONS = {
    "overview": "dashboard",
    "forecast": "query_stats",
    "inventory": "inventory_2",
    "robotics": "precision_manufacturing",
    "explainability": "psychology",
    "backtesting": "history",
    "scenario": "tune",
}

@st.cache_resource(show_spinner="Running Nexar pipeline...")
def get_pipeline_data() -> dict:
    from src.pipeline import run_pipeline
    from src.utils.config import load_config

    return run_pipeline(load_config(), include_backtesting=False, include_shap=False)


@st.cache_resource(show_spinner="Running backtesting analysis...")
def get_backtesting_data() -> dict:
    from src.pipeline import run_pipeline
    from src.utils.config import load_config

    return run_pipeline(load_config(), include_backtesting=True, include_shap=False)


@st.cache_resource(show_spinner="Computing explainability artifacts...")
def get_explainability_data() -> dict:
    from src.pipeline import run_pipeline
    from src.utils.config import load_config

    return run_pipeline(load_config(), include_backtesting=False, include_shap=True)


def apply_page_config(page_title: str) -> None:
    st.set_page_config(page_title=page_title, page_icon="■", layout="wide", initial_sidebar_state="expanded")
    theme = _get_theme_palette()
    st.session_state.setdefault("theme_mode", "dark")
    css = Template(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root {
            --bg: $bg;
            --surface: $surface;
            --surface-2: $surface_2;
            --line: $line;
            --line-strong: $line_strong;
            --text: $text;
            --muted: $muted;
            --muted-2: $muted_2;
            --accent: $accent;
            --accent-2: $accent_2;
            --accent-soft: $accent_soft;
            --good: $good;
            --warn: $warn;
            --bad: $bad;
            --shadow: $shadow;
            --hero-bg: $hero_grad_a;
            --input-bg: $input_bg;
            --table-head: $table_head;
            --table-body: $table_body;
            --table-hover: $table_hover;
            --sidebar-bg: $sidebar_grad_a;
        }

        /* ── Base ────────────────────────────────────────────── */
        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
            font-size: 14px;
        }
        [data-testid="stSidebarNav"] { display: none !important; }
        .stApp { color: var(--text); background: var(--bg); }
        .block-container {
            max-width: 1480px;
            padding: 1rem 1.25rem 3rem;
        }

        /* ── Sidebar ─────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: var(--sidebar-bg);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * { color: var(--text) !important; }
        [data-testid="stSidebar"] .block-container { padding: 1.25rem 1rem 2rem; }
        [data-testid="stSidebar"] .stButton button {
            background: var(--accent);
            color: #fff !important;
            border: none;
            border-radius: 7px;
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            min-height: 2.5rem;
        }
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
            border-left: 2px solid transparent;
            border-radius: 5px;
            padding: 0.6rem 0.8rem;
            background: transparent;
            transition: background 0.12s, border-color 0.12s;
        }
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {
            background: var(--accent-soft);
            border-left-color: var(--accent);
        }
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stMultiSelect label,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stSegmentedControl label,
        [data-testid="stSidebar"] .stToggle label {
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--muted) !important;
        }

        /* ── Headings ────────────────────────────────────────── */
        .stApp h1, .stApp h2, .stApp h3 {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--text);
            letter-spacing: -0.025em;
        }
        .stApp h2 {
            font-size: 0.94rem;
            font-weight: 600;
            letter-spacing: -0.01em;
            margin-bottom: 0.5rem;
        }
        .stApp h3 { font-size: 0.88rem; font-weight: 600; }

        /* ── Page identity ───────────────────────────────────── */
        .page-kicker {
            font-size: 0.62rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.2em;
            color: var(--accent-2);
            margin-bottom: 0.4rem;
        }
        .page-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(1.75rem, 3vw, 2.5rem);
            line-height: 1.02;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin: 0 0 0.4rem;
        }
        .page-subtitle {
            max-width: 54rem;
            color: var(--muted);
            font-size: 0.91rem;
            font-weight: 400;
            line-height: 1.58;
        }

        /* ── Page shell (header block) ───────────────────────── */
        .page-shell {
            padding: 1.1rem 1.25rem;
            margin-bottom: 1.25rem;
            background: var(--hero-bg);
            border: 1px solid var(--line);
            border-radius: 11px;
        }

        /* ── Hero badges (contextual data pills) ─────────────── */
        .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.75rem;
        }
        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.26rem 0.6rem;
            border-radius: 4px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.02);
            color: var(--muted);
            font-size: 0.7rem;
            font-weight: 500;
            transition: border-color 0.12s;
        }
        .hero-badge:hover { border-color: var(--line-strong); }
        .hero-badge-dot {
            width: 0.38rem; height: 0.38rem;
            border-radius: 50%;
            background: var(--accent);
            flex-shrink: 0;
        }

        /* ── Glass card (contextual side panels) ─────────────── */
        .hero-side { display: grid; gap: 0.75rem; }
        .glass-card {
            background: var(--surface-2);
            border: 1px solid var(--line);
            border-radius: 9px;
            padding: 0.85rem 1rem;
            transition: border-color 0.12s;
        }
        .glass-card:hover { border-color: var(--line-strong); }
        .glass-label {
            color: var(--muted);
            font-size: 0.61rem;
            font-weight: 600;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }
        .glass-value {
            margin-top: 0.25rem;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
        }

        /* ── Page meta ───────────────────────────────────────── */
        .page-meta-bottom {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.62rem;
            color: var(--muted-2);
            letter-spacing: 0.03em;
        }

        /* ── Panel system ────────────────────────────────────── */
        .panel {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 11px;
            padding: 1rem 1.1rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 6px var(--shadow);
            transition: border-color 0.12s;
        }
        .panel:hover { border-color: var(--line-strong); }
        .panel-tight { padding: 0.75rem 0.9rem; }
        .panel-hero {
            background: var(--hero-bg);
            border-left: 3px solid var(--accent);
        }

        /* ── Metric cards ────────────────────────────────────── */
        .metric-panel {
            position: relative;
            overflow: hidden;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 11px;
            padding: 1rem 1.1rem 0.9rem 1.15rem;
            min-height: 116px;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 5px var(--shadow);
            transition: border-color 0.12s;
        }
        .metric-panel:hover { border-color: var(--line-strong); }
        .metric-panel::before {
            content: "";
            position: absolute;
            left: 0; top: 0; bottom: 0;
            width: 3px;
            border-radius: 3px 0 0 3px;
            background: var(--accent);
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.64rem;
            font-weight: 600;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }
        .metric-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(1.6rem, 2.4vw, 2.15rem);
            line-height: 1.06;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-top: 0.3rem;
            color: var(--text);
        }
        .metric-detail {
            margin-top: 0.45rem;
            color: var(--muted);
            font-size: 0.79rem;
            line-height: 1.45;
        }

        /* ── Body text in panels ─────────────────────────────── */
        .note {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.62;
        }

        /* ── Entity tags (SKU labels) ────────────────────────── */
        .tag {
            display: inline-flex;
            align-items: center;
            padding: 0.13rem 0.4rem;
            margin-right: 0.35rem;
            border: 1px solid var(--line-strong);
            border-radius: 4px;
            background: var(--accent-soft);
            color: var(--accent-2);
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.64rem;
            letter-spacing: 0.02em;
            font-weight: 500;
        }

        /* ── Brand lockup ────────────────────────────────────── */
        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 0.72rem;
            margin-bottom: 1rem;
        }
        .brand-mark {
            width: 36px; height: 36px;
            border-radius: 9px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            flex-shrink: 0;
        }
        .brand-mark img { width: 100%; height: 100%; display: block; }
        .brand-name {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.98rem;
            line-height: 1;
            letter-spacing: -0.02em;
            font-weight: 700;
        }
        .brand-sub {
            color: var(--muted);
            font-size: 0.6rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-weight: 600;
            margin-top: 0.16rem;
        }

        /* ── Sidebar KPI cards ───────────────────────────────── */
        .sidebar-kpi {
            background: var(--surface-2);
            border: 1px solid var(--line);
            border-radius: 7px;
            padding: 0.7rem 0.85rem;
            margin: 0.25rem 0 0.65rem;
            transition: border-color 0.12s;
        }
        .sidebar-kpi:hover { border-color: var(--line-strong); }
        .sidebar-kpi strong {
            display: block;
            color: var(--text);
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
            letter-spacing: 0.01em;
        }
        .sidebar-kpi span {
            color: var(--muted);
            font-size: 0.6rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 600;
        }

        /* ── Status badge ────────────────────────────────────── */
        .security-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.18rem 0.5rem;
            background: var(--accent-soft);
            border: 1px solid rgba(59,130,246,0.2);
            border-radius: 4px;
            color: var(--accent-2);
            font-size: 0.58rem;
            font-weight: 600;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        /* ── Risk badges ─────────────────────────────────────── */
        .risk-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.15rem 0.44rem;
            border-radius: 4px;
            font-size: 0.61rem;
            font-weight: 600;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }
        .risk-critical { background: rgba(239,68,68,0.12); color: #f87171; border: 1px solid rgba(239,68,68,0.2); }
        .risk-high     { background: rgba(249,115,22,0.12); color: #fb923c; border: 1px solid rgba(249,115,22,0.2); }
        .risk-medium   { background: rgba(234,179,8,0.1);   color: #eab308; border: 1px solid rgba(234,179,8,0.18); }
        .risk-low      { background: rgba(34,197,94,0.1);   color: #22c55e; border: 1px solid rgba(34,197,94,0.18); }

        /* ── Chart legend ────────────────────────────────────── */
        .legend-rail {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem 0.85rem;
            align-items: center;
            margin-bottom: 0.3rem;
        }
        .legend-item {
            display: inline-flex;
            align-items: center;
            gap: 0.36rem;
            color: var(--muted);
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }
        .legend-swatch {
            width: 8px; height: 8px;
            border-radius: 2px;
        }

        /* ── Status rail (KPI row inside panels) ─────────────── */
        .status-rail {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(118px, 1fr));
            gap: 0.55rem;
        }
        .status-card {
            background: var(--surface-2);
            border: 1px solid var(--line);
            border-radius: 7px;
            padding: 0.65rem 0.78rem;
            transition: border-color 0.12s;
        }
        .status-card:hover { border-color: var(--line-strong); }
        .status-card-label {
            color: var(--muted);
            font-size: 0.61rem;
            font-weight: 600;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }
        .status-card-value {
            margin-top: 0.22rem;
            color: var(--text);
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.95rem;
            font-weight: 500;
        }

        /* ── Table captions ──────────────────────────────────── */
        .table-note {
            color: var(--muted);
            font-size: 0.68rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        /* ── DataFrames ──────────────────────────────────────── */
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }
        div[data-testid="stDataFrame"] [role="table"] { color: var(--text); }
        div[data-testid="stDataFrame"] thead tr th {
            background: var(--table-head) !important;
            color: var(--muted) !important;
            font-size: 0.63rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
            font-weight: 600 !important;
            border-bottom: 1px solid var(--line) !important;
        }
        div[data-testid="stDataFrame"] tbody tr td {
            background: var(--table-body) !important;
            border-bottom: 1px solid var(--line) !important;
            font-size: 0.79rem !important;
        }
        div[data-testid="stDataFrame"] tbody tr:hover td { background: var(--table-hover) !important; }

        /* ── Form elements ───────────────────────────────────── */
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div,
        .stTextInput input,
        .stNumberInput input {
            background: var(--input-bg);
            border: 1px solid var(--line);
            color: var(--text);
            border-radius: 6px;
        }
        .stButton button { border-radius: 7px; border: 1px solid var(--line); }
        .stSlider [data-baseweb="slider"] { padding-top: 0.2rem; }
        .stSlider [role="slider"] { background: var(--accent); }
        .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"] { background: var(--accent-soft); }
        .stSegmentedControl [data-baseweb="button-group"] {
            background: var(--input-bg);
            border: 1px solid var(--line);
            border-radius: 7px;
            padding: 0.18rem;
        }
        .stSegmentedControl button[data-baseweb="button"] {
            color: var(--muted);
            font-size: 0.67rem;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            font-weight: 600;
            border-radius: 4px;
        }
        .stSegmentedControl button[aria-pressed="true"] {
            background: var(--accent) !important;
            color: #f4f8ff !important;
        }
        .stToggle {
            background: rgba(255,255,255,0.025);
            border: 1px solid var(--line);
            border-radius: 7px;
            padding: 0.45rem 0.75rem;
        }
        .stToggle [data-baseweb="checkbox"] > label { color: var(--muted) !important; }
        .stCaption { color: var(--muted) !important; font-size: 0.8rem !important; }
        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 9px;
            padding: 0.7rem 0.85rem;
        }

        /* ── Scrollbar ───────────────────────────────────────── */
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--line-strong); border-radius: 99px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--muted-2); }

        /* ── Layout helpers ──────────────────────────────────── */
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            margin-bottom: 0 !important;
        }
        div[data-testid="stHorizontalBlock"] {
            align-items: flex-start !important;
            gap: 1.25rem !important;
        }
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMarkdownContainer"]),
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stPlotlyChart"]),
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stDataFrame"]) {
            margin-bottom: 0.3rem;
        }
        div[data-testid="stPlotlyChart"] { margin-top: 0.4rem; margin-bottom: 0.1rem; }

        /* ── Responsive ──────────────────────────────────────── */
        @media (max-width: 1200px) {
            .block-container { padding-left: 0.9rem; padding-right: 0.9rem; }
        }
        @media (max-width: 768px) {
            .block-container { padding-top: 0.5rem; padding-left: 0.65rem; padding-right: 0.65rem; }
            .page-title { font-size: 1.65rem; }
            .page-shell, .panel, .metric-panel { border-radius: 9px; }
            div[data-testid="stHorizontalBlock"] { gap: 0.75rem !important; }
        }
        </style>
        """
    ).substitute(theme)
    st.markdown(css, unsafe_allow_html=True)


def render_sidebar(active_key: str, data: Optional[dict] = None) -> None:
    mark_uri = _logo_data_uri()
    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand-lockup">
                <div class="brand-mark">
                    <img src="{mark_uri}" alt="Nexar mark" />
                </div>
                <div>
                    <div class="brand-name">Nexar</div>
                    <div class="brand-sub">Supply Chain Intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if data is not None:
            future_df = data["future_df"]
            inventory_df = data["inventory_df"]
            st.markdown(
                f"""
                <div class="sidebar-kpi">
                    <span>Projected Demand</span>
                    <strong>{int(future_df['forecast'].sum()):,} units</strong>
                </div>
                <div class="sidebar-kpi">
                    <span>Restock Watchlist</span>
                    <strong>{int(inventory_df['reorder_needed'].sum())} SKUs flagged</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        _page_link("streamlit_app.py", PAGE_TITLES["overview"], ":material/dashboard:", active_key == "overview")
        _page_link("pages/1_Forecast_Explorer.py", PAGE_TITLES["forecast"], ":material/query_stats:", active_key == "forecast")
        _page_link("pages/2_Inventory_Decisions.py", PAGE_TITLES["inventory"], ":material/inventory_2:", active_key == "inventory")
        _page_link("pages/3_Robotics_Simulation.py", PAGE_TITLES["robotics"], ":material/precision_manufacturing:", active_key == "robotics")
        _page_link("pages/4_Explainability.py", PAGE_TITLES["explainability"], ":material/psychology:", active_key == "explainability")
        _page_link("pages/5_Backtesting.py", PAGE_TITLES["backtesting"], ":material/history:", active_key == "backtesting")
        _page_link("pages/6_Scenario_Simulator.py", PAGE_TITLES["scenario"], ":material/tune:", active_key == "scenario")
        st.divider()
        st.caption(f"Theme: {st.session_state.get('theme_mode', 'dark').title()}")


def render_header(kicker: str, title: str, subtitle: str) -> None:
    left, right = st.columns([1.7, 0.9], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="page-shell">
                <div class="page-kicker">{kicker}</div>
                <div class="page-title">{title}</div>
                <div class="page-subtitle">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        theme_toggle = st.toggle(
            "Light theme",
            value=st.session_state.get("theme_mode", "dark") == "light",
            key="theme_mode_toggle",
            help="Switch between dark and light UI modes.",
        )
        desired_theme = "light" if theme_toggle else "dark"
        if desired_theme != st.session_state.get("theme_mode"):
            st.session_state["theme_mode"] = desired_theme
            st.rerun()


def metric_panel(label: str, value: str, detail: str) -> str:
    return (
        '<div class="metric-panel">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-detail">{detail}</div>'
        "</div>"
    )


def style_plotly(fig, height: int = 320):
    theme = _get_theme_palette()
    tick_font = {"color": theme["plot_tick"], "size": 11, "family": "IBM Plex Mono, monospace"}
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=8, t=24, b=8),
        paper_bgcolor=theme["plot_bg"],
        plot_bgcolor=theme["plot_bg"],
        font={"family": "IBM Plex Sans, sans-serif", "color": theme["plot_text"], "size": 12},
        legend={
            "font": {"color": theme["muted"], "size": 10, "family": "IBM Plex Sans, sans-serif"},
            "bgcolor": "rgba(255,255,255,0)",
            "borderwidth": 0,
            "orientation": "h",
            "x": 0,
            "y": 1.06,
            "itemclick": False,
            "itemdoubleclick": False,
        },
        hoverlabel={
            "bgcolor": theme["surface_2"],
            "bordercolor": theme["line_strong"],
            "font": {"family": "IBM Plex Sans, sans-serif", "size": 12, "color": theme["text"]},
        },
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=theme["plot_line"],
        tickfont=tick_font,
        title_font={"color": theme["muted"], "size": 11},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=theme["plot_grid"],
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont=tick_font,
        title_font={"color": theme["muted"], "size": 11},
    )
    return fig


def compact_legend(items: list[tuple[str, str]]) -> str:
    parts = []
    for label, color in items:
        parts.append(
            f"<div class='legend-item'>"
            f"<span class='legend-swatch' style='background:{color};border-radius:2px;'></span>"
            f"{label}"
            f"</div>"
        )
    return "<div class='legend-rail'>" + "".join(parts) + "</div>"


def status_rail(items: list[tuple[str, str]]) -> str:
    parts = []
    for label, value in items:
        parts.append(
            f"<div class='status-card'><div class='status-card-label'>{label}</div><div class='status-card-value'>{value}</div></div>"
        )
    return "<div class='status-rail'>" + "".join(parts) + "</div>"


def dense_dataframe(df, height: int = 320):
    st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def _logo_data_uri() -> str:
    path = Path(__file__).resolve().parents[2] / "assets" / "demandsense_mark.svg"
    if not path.exists():
        fallback = (
            "<svg width='96' height='96' viewBox='0 0 96 96' fill='none' xmlns='http://www.w3.org/2000/svg'>"
            "<rect width='96' height='96' rx='22' fill='url(#bg)'/>"
            "<rect x='1' y='1' width='94' height='94' rx='21' stroke='rgba(255,255,255,0.1)' stroke-width='1.5'/>"
            "<polygon points='48,17 71.4,30.5 71.4,57.5 48,71 24.6,57.5 24.6,30.5' fill='none' stroke='white' stroke-width='2' stroke-linejoin='round' opacity='0.65'/>"
            "<polygon points='48,30 60.1,37 60.1,51 48,58 35.9,51 35.9,37' fill='rgba(255,255,255,0.06)' stroke='white' stroke-width='1.5' stroke-linejoin='round' opacity='0.32'/>"
            "<line x1='48' y1='44' x2='48' y2='17' stroke='white' stroke-width='1.2' opacity='0.22'/>"
            "<line x1='48' y1='44' x2='71.4' y2='57.5' stroke='white' stroke-width='1.2' opacity='0.22'/>"
            "<line x1='48' y1='44' x2='24.6' y2='57.5' stroke='white' stroke-width='1.2' opacity='0.22'/>"
            "<circle cx='48' cy='44' r='12' fill='#2563eb' fill-opacity='0.18'/>"
            "<circle cx='48' cy='44' r='7' fill='#60a5fa' opacity='0.95'/>"
            "<circle cx='48' cy='17' r='3.8' fill='white' opacity='0.88'/>"
            "<circle cx='71.4' cy='30.5' r='3.8' fill='white' opacity='0.88'/>"
            "<circle cx='24.6' cy='30.5' r='3.8' fill='white' opacity='0.88'/>"
            "<circle cx='71.4' cy='57.5' r='2.8' fill='white' opacity='0.42'/>"
            "<circle cx='48' cy='71' r='2.8' fill='white' opacity='0.42'/>"
            "<circle cx='24.6' cy='57.5' r='2.8' fill='white' opacity='0.42'/>"
            "<defs><linearGradient id='bg' x1='0' y1='0' x2='96' y2='96' gradientUnits='userSpaceOnUse'>"
            "<stop offset='0' stop-color='#0b1f47'/><stop offset='0.5' stop-color='#102d6b'/><stop offset='1' stop-color='#1a4491'/>"
            "</linearGradient></defs></svg>"
        )
        encoded = base64.b64encode(fallback.encode("utf-8")).decode("utf-8")
        return f"data:image/svg+xml;base64,{encoded}"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


def segmented_control(label: str, options: list, default=None, key: Optional[str] = None):
    if hasattr(st, "segmented_control"):
        return st.segmented_control(label, options=options, default=default, key=key)
    default_index = options.index(default) if default in options else 0
    return st.radio(label, options=options, index=default_index, horizontal=True, key=key)


def _page_link(page: str, label: str, icon: str, disabled: bool) -> None:
    if hasattr(st, "page_link"):
        st.page_link(page, label=label, icon=icon, disabled=disabled)
    else:
        st.markdown(f"`{label}`")


def _get_theme_palette() -> dict[str, str]:
    mode = st.session_state.get("theme_mode", "dark")
    if mode == "light":
        # Cloud White — clean, high-contrast light theme
        return {
            "bg": "#eef3f9",           # page background: slightly blue-tinted off-white
            "bg_soft": "#f5f8fc",
            "surface": "#ffffff",      # panels: pure white — clear lift above bg
            "surface_2": "#f4f8fd",    # nested/hover surfaces
            "surface_3": "#eaf0f8",
            "line": "rgba(15, 35, 75, 0.1)",
            "line_strong": "rgba(15, 35, 75, 0.18)",
            "text": "#0d1828",         # near-black with blue undertone
            "muted": "#52698a",        # readable but clearly secondary
            "muted_2": "#8ba0bc",
            "accent": "#1d4ed8",       # blue-700 — authoritative
            "accent_2": "#2563eb",     # blue-600
            "accent_soft": "rgba(29, 78, 216, 0.07)",
            "good": "#16a34a",
            "warn": "#c2410c",
            "bad": "#b91c1c",
            "shadow": "rgba(10, 25, 60, 0.07)",
            "card_glow": "rgba(29, 78, 216, 0.06)",
            "app_grad_a": "rgba(29, 78, 216, 0.04)",
            "app_grad_b": "rgba(14, 165, 233, 0.03)",
            "app_grad_c": "#f5f8fc",
            "sidebar_grad_a": "#f0f5fb",  # sidebar: slightly different from page bg
            "sidebar_grad_b": "#e8f0f9",
            "hero_grad_a": "#edf4ff",     # hero sections: clean blue-tinted white
            "hero_grad_b": "#e8f1fd",
            "hero_grad_c": "#e2ecfb",
            "panel_grad_a": "#ffffff",
            "panel_grad_b": "#ffffff",
            "metric_grad_a": "#ffffff",
            "metric_grad_b": "#ffffff",
            "input_bg": "#ffffff",
            "table_head": "#f0f5fb",
            "table_body": "#ffffff",
            "table_hover": "#f5f9ff",
            "plot_bg": "rgba(0,0,0,0)",
            "plot_text": "#0d1828",
            "plot_grid": "rgba(15, 35, 75, 0.07)",
            "plot_line": "rgba(15, 35, 75, 0.12)",
            "plot_tick": "#52698a",
        }
    # Carbon Navy — precise dark theme
    return {
        "bg": "#080e1b",           # base: very dark blue-black
        "bg_soft": "#0c1220",
        "surface": "#0f1825",      # panels: clearly elevated above bg
        "surface_2": "#141f2e",    # hover/nested surfaces
        "surface_3": "#192638",
        "line": "rgba(75, 125, 195, 0.15)",
        "line_strong": "rgba(105, 160, 225, 0.26)",
        "text": "#dce8f8",         # cool near-white
        "muted": "#6788a8",        # readable secondary text
        "muted_2": "#3d5a78",
        "accent": "#3b82f6",       # blue-500 — airy and readable
        "accent_2": "#60a5fa",     # blue-400
        "accent_soft": "rgba(59, 130, 246, 0.1)",
        "good": "#22c55e",
        "warn": "#f97316",
        "bad": "#ef4444",
        "shadow": "rgba(0, 0, 0, 0.3)",
        "card_glow": "rgba(59, 130, 246, 0.14)",
        "app_grad_a": "rgba(59, 130, 246, 0.07)",
        "app_grad_b": "rgba(14, 165, 233, 0.05)",
        "app_grad_c": "#060c18",
        "sidebar_grad_a": "#070c18",   # sidebar: fractionally darker than bg
        "sidebar_grad_b": "#070c18",
        "hero_grad_a": "#101c30",      # hero sections: slight blue lift
        "hero_grad_b": "#0c1628",
        "hero_grad_c": "#091320",
        "panel_grad_a": "#0f1825",     # same as surface — flat, clean panels
        "panel_grad_b": "#0f1825",
        "metric_grad_a": "#0f1825",
        "metric_grad_b": "#0f1825",
        "input_bg": "#0c1724",
        "table_head": "#0c1828",
        "table_body": "#0d1a26",
        "table_hover": "#121f2e",
        "plot_bg": "rgba(0,0,0,0)",
        "plot_text": "#bfd0e8",
        "plot_grid": "rgba(90, 130, 175, 0.12)",
        "plot_line": "rgba(90, 130, 175, 0.18)",
        "plot_tick": "#6788a8",
    }
