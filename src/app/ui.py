"""Shared Streamlit UI helpers for DemandSense-RX."""

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

@st.cache_resource(show_spinner="Running DemandSense-RX pipeline...")
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
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');
        :root{
            --bg:$bg;
            --bg-soft:$bg_soft;
            --surface:$surface;
            --surface-2:$surface_2;
            --surface-3:$surface_3;
            --line:$line;
            --line-strong:$line_strong;
            --text:$text;
            --muted:$muted;
            --muted-2:$muted_2;
            --accent:$accent;
            --accent-2:$accent_2;
            --accent-soft:$accent_soft;
            --good:$good;
            --warn:$warn;
            --bad:$bad;
            --shadow:$shadow;
            --card-glow:$card_glow;
            --app-grad-a:$app_grad_a;
            --app-grad-b:$app_grad_b;
            --app-grad-c:$app_grad_c;
            --sidebar-grad-a:$sidebar_grad_a;
            --sidebar-grad-b:$sidebar_grad_b;
            --hero-grad-a:$hero_grad_a;
            --hero-grad-b:$hero_grad_b;
            --hero-grad-c:$hero_grad_c;
            --panel-grad-a:$panel_grad_a;
            --panel-grad-b:$panel_grad_b;
            --metric-grad-a:$metric_grad_a;
            --metric-grad-b:$metric_grad_b;
            --input-bg:$input_bg;
            --table-head:$table_head;
            --table-body:$table_body;
            --table-hover:$table_hover;
        }
        html, body, [class*="css"] { font-family:'IBM Plex Sans', sans-serif; }
        /* Hide native Streamlit multipage nav — custom nav in sidebar replaces it */
        [data-testid="stSidebarNav"] { display: none !important; }
        .stApp {
            color: var(--text);
            background:
                radial-gradient(circle at top left, var(--app-grad-a), transparent 34%),
                radial-gradient(circle at top right, var(--app-grad-b), transparent 30%),
                linear-gradient(180deg, var(--app-grad-c) 0%, var(--bg) 100%);
        }
        .block-container {
            max-width: 1500px;
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 1.1rem;
            padding-right: 1.1rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--sidebar-grad-a) 0%, var(--sidebar-grad-b) 100%);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * { color: var(--text) !important; }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
            padding-left: 0.9rem;
            padding-right: 0.9rem;
        }
        [data-testid="stSidebar"] .stButton button {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
            color: #fcfcff !important;
            border: none;
            border-radius: 999px;
            font-size: 0.69rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            min-height: 2.8rem;
            box-shadow: 0 18px 38px var(--card-glow);
        }
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
            border-left: 2px solid transparent;
            border-radius: 14px;
            padding: 0.75rem 0.9rem;
            background: transparent;
            transition: background-color 0.18s ease, border-color 0.18s ease, transform 0.18s ease;
        }
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {
            background: var(--accent-soft);
            border-left-color: rgba(99,102,241,0.6);
            transform: translateX(2px);
        }
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stMultiSelect label,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stSegmentedControl label,
        [data-testid="stSidebar"] .stToggle label {
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--muted) !important;
        }
        .stApp h1, .stApp h2, .stApp h3 {
            font-family:'Space Grotesk', sans-serif;
            color: var(--text);
            letter-spacing: -0.03em;
        }
        .page-kicker {
            font-size: 0.67rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.24em;
            color: var(--accent-2);
            margin-bottom: 0.5rem;
        }
        .page-title {
            font-size: clamp(2.35rem, 5vw, 4.1rem);
            line-height: 0.95;
            font-weight: 700;
            margin: 0 0 0.45rem 0;
        }
        .page-subtitle {
            max-width: 58rem;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.55;
        }
        .hero-badges {
            display:flex;
            flex-wrap:wrap;
            gap:0.65rem;
            margin-top:1rem;
        }
        .hero-badge {
            display:inline-flex;
            align-items:center;
            gap:0.5rem;
            padding:0.44rem 0.9rem;
            border-radius:999px;
            border:1px solid var(--line);
            background:rgba(255,255,255,0.04);
            color:var(--text);
            font-size:0.74rem;
            font-weight:600;
            letter-spacing:0.01em;
            transition: border-color 0.2s, background 0.2s;
        }
        .hero-badge:hover {
            border-color: var(--line-strong);
            background:rgba(255,255,255,0.07);
        }
        .hero-badge-dot {
            width:0.52rem;
            height:0.52rem;
            border-radius:999px;
            background:var(--accent);
            box-shadow:0 0 10px var(--card-glow);
            flex-shrink:0;
        }
        .hero-side {
            display:grid;
            gap:0.9rem;
        }
        .glass-card {
            background:rgba(255,255,255,0.04);
            border:1px solid var(--line);
            border-radius:18px;
            padding:1rem 1.1rem;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 2px 12px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,0.06);
            transition: border-color 0.2s ease, background 0.2s ease;
        }
        .glass-card:hover {
            background:rgba(255,255,255,0.06);
            border-color: var(--line-strong);
        }
        .glass-label {
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }
        .glass-value {
            margin-top:0.32rem;
            font-family:'Space Grotesk', sans-serif;
            font-size:1.2rem;
            font-weight:700;
        }
        .page-meta {
            display:flex;
            flex-direction:column;
            align-items:flex-start;
            gap:0.4rem;
            text-align:left;
        }
        .page-meta-top {
            font-size: 0.66rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--muted);
        }
        .page-meta-bottom {
            font-family:'IBM Plex Mono', monospace;
            font-size: 0.66rem;
            color: var(--muted-2);
            letter-spacing: 0.04em;
        }
        .panel {
            background:
                linear-gradient(180deg, var(--panel-grad-a) 0%, var(--panel-grad-b) 100%);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1.1rem 1.15rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 4px 24px var(--shadow), 0 1px 4px rgba(0,0,0,0.06);
            transition: box-shadow 0.22s ease, transform 0.22s ease, border-color 0.22s ease;
        }
        .panel:hover {
            box-shadow: 0 8px 40px var(--shadow), 0 2px 8px rgba(0,0,0,0.08);
            transform: translateY(-1px);
            border-color: var(--line-strong);
        }
        .panel-tight { padding: 0.82rem 0.95rem; }
        .panel-hero {
            background:
                linear-gradient(135deg, var(--hero-grad-a) 0%, var(--hero-grad-b) 65%, var(--hero-grad-c) 100%);
            border-left: 3px solid var(--accent);
        }
        .panel-hero:hover { transform: none; box-shadow: 0 4px 24px var(--shadow); }
        .metric-panel {
            position:relative;
            overflow:hidden;
            background: linear-gradient(160deg, var(--metric-grad-a) 0%, var(--metric-grad-b) 100%);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1.1rem 1.1rem 1rem 1.2rem;
            min-height: 148px;
            margin-bottom: 1.1rem;
            box-shadow: 0 4px 20px var(--shadow);
            transition: box-shadow 0.22s ease, transform 0.22s ease, border-color 0.22s ease;
        }
        .metric-panel:hover {
            box-shadow: 0 8px 36px var(--shadow);
            transform: translateY(-1px);
            border-color: var(--line-strong);
        }
        .metric-panel::before {
            content:"";
            position:absolute;
            left:0;
            top:0;
            bottom:0;
            width:3px;
            border-radius:3px 0 0 3px;
            background: linear-gradient(180deg, var(--accent) 0%, var(--accent-soft) 100%);
        }
        .metric-panel::after {
            content:"";
            position:absolute;
            top:-32px;
            right:-32px;
            width:100px;
            height:100px;
            border-radius:50%;
            background: radial-gradient(circle, var(--card-glow) 0%, transparent 70%);
            pointer-events:none;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
        }
        .metric-value {
            font-family:'Space Grotesk', sans-serif;
            font-size: clamp(1.9rem, 3vw, 2.7rem);
            line-height: 1.04;
            font-weight: 700;
            margin-top: 0.38rem;
            color: var(--text);
        }
        .metric-detail {
            margin-top: 0.55rem;
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.45;
        }
        .metric-chip {
            display:inline-flex;
            margin-top:0.9rem;
            padding:0.26rem 0.6rem;
            border-radius:999px;
            background:var(--accent-soft);
            border:1px solid rgba(99,102,241,0.2);
            color:var(--accent-2);
            font-size:0.68rem;
            font-weight:700;
            letter-spacing:0.06em;
        }
        .note {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.58;
        }
        .tag {
            display:inline-flex;
            align-items:center;
            padding:0.16rem 0.44rem;
            margin-right:0.38rem;
            border:1px solid var(--line-strong);
            border-radius:5px;
            background:var(--accent-soft);
            color:var(--accent-2);
            font-family:'IBM Plex Mono', monospace;
            font-size:0.67rem;
            letter-spacing:0.03em;
            font-weight:600;
        }
        .status-strip {
            display:flex;
            gap:1.5rem;
            flex-wrap:wrap;
            align-items:center;
            margin-top:0.85rem;
        }
        .status-item {
            display:flex;
            align-items:center;
            gap:0.4rem;
            color:var(--muted);
            font-size:0.7rem;
            letter-spacing:0.12em;
            text-transform:uppercase;
            font-weight:700;
        }
        .status-dot {
            width:7px;
            height:7px;
            border-radius:999px;
            background:var(--accent);
            box-shadow:0 0 12px rgba(99,102,241,0.65);
        }
        .brand-lockup {
            display:flex;
            align-items:center;
            gap:0.8rem;
            margin-bottom:0.7rem;
        }
        .brand-mark {
            width:42px;
            height:42px;
            border-radius:13px;
            display:flex;
            align-items:center;
            justify-content:center;
            overflow:hidden;
            box-shadow:0 6px 24px var(--card-glow), 0 1px 3px rgba(0,0,0,0.2);
            flex-shrink:0;
        }
        .brand-mark img { width:100%; height:100%; display:block; }
        .brand-name {
            font-family:'Space Grotesk', sans-serif;
            font-size:1.08rem;
            line-height:1;
            letter-spacing:-0.03em;
            font-weight:700;
        }
        .brand-sub {
            color: var(--muted);
            font-size:0.64rem;
            letter-spacing:0.16em;
            text-transform:uppercase;
            font-weight:700;
            margin-top:0.22rem;
        }
        .sidebar-kpi {
            background: linear-gradient(160deg, var(--panel-grad-a) 0%, var(--panel-grad-b) 100%);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
            margin: 0.3rem 0 0.7rem 0;
            transition: border-color 0.2s ease;
        }
        .sidebar-kpi:hover { border-color: var(--line-strong); }
        .sidebar-kpi strong {
            display:block;
            color:var(--text);
            font-family:'IBM Plex Mono', monospace;
            font-size:0.82rem;
            letter-spacing:0.02em;
        }
        .sidebar-kpi span {
            color:var(--muted);
            font-size:0.65rem;
            letter-spacing:0.14em;
            text-transform:uppercase;
            font-weight:700;
        }
        .security-pill {
            display:inline-flex;
            align-items:center;
            gap:0.38rem;
            padding:0.28rem 0.7rem;
            background:var(--accent-soft);
            border:1px solid rgba(99,102,241,0.25);
            border-radius:999px;
            color:var(--accent-2);
            font-size:0.61rem;
            font-weight:700;
            letter-spacing:0.14em;
            text-transform:uppercase;
        }
        .legend-rail {
            display:flex;
            flex-wrap:wrap;
            gap:0.75rem 1rem;
            align-items:center;
        }
        .legend-item {
            display:inline-flex;
            align-items:center;
            gap:0.42rem;
            color:var(--muted);
            font-size:0.68rem;
            font-weight:700;
            letter-spacing:0.12em;
            text-transform:uppercase;
        }
        .legend-swatch {
            width:10px;
            height:10px;
            border-radius:999px;
            border:1px solid rgba(255,255,255,0.14);
        }
        .status-rail {
            display:grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap:0.7rem;
        }
        .status-card {
            background:rgba(255,255,255,0.025);
            border:1px solid var(--line);
            border-radius:16px;
            padding:0.85rem 0.9rem;
            transition: border-color 0.2s ease, background 0.2s ease;
        }
        .status-card:hover {
            background:rgba(255,255,255,0.045);
            border-color: var(--line-strong);
        }
        .status-card-label {
            color:var(--muted);
            font-size:0.64rem;
            font-weight:700;
            letter-spacing:0.14em;
            text-transform:uppercase;
        }
        .status-card-value {
            margin-top:0.28rem;
            color:#f2f5ff;
            font-family:'IBM Plex Mono', monospace;
            font-size:1rem;
        }
        .table-note {
            color:var(--muted);
            font-size:0.72rem;
            letter-spacing:0.08em;
            text-transform:uppercase;
        }
        div[data-testid="stDataFrame"] {
            border:1px solid var(--line);
            border-radius:18px;
            overflow:hidden;
            background:var(--table-body);
        }
        div[data-testid="stDataFrame"] [role="table"] { color:var(--text); }
        div[data-testid="stDataFrame"] thead tr th {
            background:var(--table-head) !important;
            color:var(--muted) !important;
            font-size:0.67rem !important;
            text-transform:uppercase !important;
            letter-spacing:0.12em !important;
            border-bottom:1px solid var(--line) !important;
        }
        div[data-testid="stDataFrame"] tbody tr td {
            background:var(--table-body) !important;
            border-bottom:1px solid var(--line) !important;
            font-size:0.77rem !important;
        }
        div[data-testid="stDataFrame"] tbody tr:hover td { background:var(--table-hover) !important; }
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div,
        .stTextInput input,
        .stNumberInput input {
            background:var(--input-bg);
            border:1px solid var(--line);
            color:var(--text);
            border-radius:16px;
        }
        .stButton button {
            border-radius:999px;
            border:1px solid var(--line);
        }
        .stSlider [data-baseweb="slider"] { padding-top:0.25rem; }
        .stSlider [role="slider"] { background: var(--accent); }
        .stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] { background: rgba(99,102,241,0.18); }
        .stSegmentedControl [data-baseweb="button-group"] {
            background:var(--input-bg);
            border:1px solid var(--line);
            border-radius:16px;
            padding:0.2rem;
        }
        .stSegmentedControl button[data-baseweb="button"] {
            color:var(--muted);
            font-size:0.7rem;
            letter-spacing:0.12em;
            text-transform:uppercase;
            font-weight:700;
            border-radius:4px;
        }
        .stSegmentedControl button[aria-pressed="true"] {
            background:linear-gradient(135deg, rgba(79,70,229,0.96) 0%, rgba(129,140,248,0.96) 100%) !important;
            color:#f7f6ff !important;
        }
        .stToggle {
            background:rgba(255,255,255,0.04);
            border:1px solid var(--line);
            border-radius:18px;
            padding:0.55rem 0.85rem;
        }
        .stToggle [data-baseweb="checkbox"] > label { color: var(--muted) !important; }
        .stCaption { color: var(--muted) !important; }
        div[data-testid="stMetric"] {
            background:linear-gradient(180deg, var(--metric-grad-a) 0%, var(--metric-grad-b) 100%);
            border:1px solid var(--line);
            border-radius:20px;
            padding:0.75rem 0.9rem;
        }
        /* Custom scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--line-strong); border-radius: 99px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--muted-2); }
        /* Page shell shimmer */
        .page-shell {
            position: relative;
            overflow: hidden;
            padding: 1.4rem;
            margin-bottom: 1.15rem;
            background:
                radial-gradient(circle at top right, var(--card-glow), transparent 42%),
                linear-gradient(135deg, var(--hero-grad-a) 0%, var(--hero-grad-b) 58%, var(--hero-grad-c) 100%);
            border: 1px solid var(--line);
            border-radius: 24px;
            box-shadow: 0 4px 32px var(--shadow);
        }
        .page-shell::before {
            content:"";
            position:absolute;
            inset:0;
            background: linear-gradient(105deg, rgba(255,255,255,0.03) 0%, transparent 50%);
            pointer-events:none;
        }
        @media (max-width: 1200px) {
            .block-container { padding-left: 0.9rem; padding-right: 0.9rem; }
            .page-shell { padding: 1.15rem; }
        }
        /* Ensure vertical stacking inside columns always has breathing room */
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            margin-bottom: 0 !important;
        }
        div[data-testid="stHorizontalBlock"] {
            align-items: flex-start !important;
            gap: 1.1rem !important;
        }
        @media (max-width: 768px) {
            .block-container { padding-top: 0.6rem; padding-left: 0.7rem; padding-right: 0.7rem; }
            .page-title { line-height: 1.02; }
            .page-shell { border-radius: 20px; }
            .panel, .metric-panel, .sidebar-kpi { border-radius: 18px; }
            div[data-testid="stHorizontalBlock"] { gap: 0.8rem !important; }
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
                    <img src="{mark_uri}" alt="DemandSense-RX mark" />
                </div>
                <div>
                    <div class="brand-name">DemandSense-RX</div>
                    <div class="brand-sub">Forecasting and Fulfilment</div>
                </div>
            </div>
            <div class="security-pill">Brand System Active</div>
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
        st.markdown(
            """
            <div class="status-strip" style="margin-top:0.25rem; margin-bottom:0.85rem;">
                <div class="status-item"><span class="status-dot"></span>Live Workspace</div>
                <div class="status-item">Adaptive Layout</div>
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
        st.button("Run New Scenario", width="stretch")
        st.caption(f"Theme: {st.session_state.get('theme_mode', 'dark').title()} · Responsive dashboard")


def render_header(kicker: str, title: str, subtitle: str) -> None:
    left, right = st.columns([1.7, 0.9], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="page-shell">
                <div class="page-kicker">{kicker}</div>
                <div class="page-title">{title}</div>
                <div class="page-subtitle">{subtitle}</div>
                <div class="hero-badges">
                    <div class="hero-badge"><span class="hero-badge-dot"></span>Brand-led UI system</div>
                    <div class="hero-badge"><span class="hero-badge-dot"></span>Mobile to desktop ready</div>
                    <div class="hero-badge"><span class="hero-badge-dot"></span>Real-time planning signals</div>
                </div>
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

        st.markdown(
            """
            <div class="hero-side">
                <div class="glass-card">
                    <div class="glass-label">Interface Mode</div>
                    <div class="glass-value">Precision Dashboard</div>
                    <div class="page-meta-bottom" style="margin-top:0.35rem;">Balanced for touch, laptop, and widescreen workflows</div>
                </div>
                <div class="glass-card">
                    <div class="glass-label">Operational Status</div>
                    <div class="glass-value">Live planning fabric</div>
                    <div class="page-meta-bottom" style="margin-top:0.35rem;">Forecast, inventory, and warehouse views aligned on one design language</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def metric_panel(label: str, value: str, detail: str) -> str:
    return (
        '<div class="metric-panel">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-detail">{detail}</div>'
        '<div class="metric-chip">DemandSense-RX</div>'
        "</div>"
    )


def style_plotly(fig, height: int = 320):
    theme = _get_theme_palette()
    plotly_theme = {
        "paper_bgcolor": theme["plot_bg"],
        "plot_bgcolor": theme["plot_bg"],
        "font": {"family": "IBM Plex Sans, sans-serif", "color": theme["plot_text"]},
        "legend": {"font": {"color": theme["muted"]}},
    }
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), **plotly_theme)
    fig.update_xaxes(showgrid=True, gridcolor=theme["plot_grid"], zeroline=False, linecolor=theme["plot_line"], tickfont={"color": theme["plot_tick"]})
    fig.update_yaxes(showgrid=True, gridcolor=theme["plot_grid"], zeroline=False, linecolor=theme["plot_line"], tickfont={"color": theme["plot_tick"]})
    return fig


def compact_legend(items: list[tuple[str, str]]) -> str:
    parts = []
    for label, color in items:
        parts.append(
            f"<div class='legend-item'><span class='legend-swatch' style='background:{color};'></span>{label}</div>"
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
    st.dataframe(df, width="stretch", hide_index=True, height=height)


def _logo_data_uri() -> str:
    path = Path(__file__).resolve().parents[2] / "assets" / "demandsense_mark.svg"
    if not path.exists():
        fallback = (
            "<svg width='96' height='96' viewBox='0 0 96 96' fill='none' xmlns='http://www.w3.org/2000/svg'>"
            "<rect width='96' height='96' rx='20' fill='url(#bg)'/>"
            "<rect x='1' y='1' width='94' height='94' rx='19' stroke='rgba(255,255,255,0.14)' stroke-width='1.5'/>"
            "<path d='M15 50 L27 50 L33 29 L41 69 L47 35 L53 59 L61 50 L81 50' stroke='white' stroke-width='3.2' stroke-linecap='round' stroke-linejoin='round' opacity='0.9'/>"
            "<circle cx='47' cy='35' r='5.2' fill='#c7d2fe'/>"
            "<circle cx='47' cy='35' r='9.5' fill='#818cf8' fill-opacity='0.18'/>"
            "<circle cx='15' cy='50' r='2.6' fill='white' fill-opacity='0.42'/>"
            "<circle cx='81' cy='50' r='2.6' fill='white' fill-opacity='0.42'/>"
            "<defs><linearGradient id='bg' x1='0' y1='0' x2='96' y2='96' gradientUnits='userSpaceOnUse'>"
            "<stop offset='0' stop-color='#1e1b4b'/><stop offset='0.48' stop-color='#2d2a80'/><stop offset='1' stop-color='#4338ca'/>"
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
        return {
            "bg": "#f1f5fa",
            "bg_soft": "#f8fbff",
            "surface": "#ffffff",
            "surface_2": "#f4f7fb",
            "surface_3": "#edf2f8",
            "line": "rgba(15, 30, 60, 0.09)",
            "line_strong": "rgba(15, 30, 60, 0.16)",
            "text": "#0f172a",
            "muted": "#64748b",
            "muted_2": "#94a3b8",
            "accent": "#4f46e5",
            "accent_2": "#7c73f8",
            "accent_soft": "rgba(79, 70, 229, 0.08)",
            "good": "#059669",
            "warn": "#d97706",
            "bad": "#dc2626",
            "shadow": "rgba(15, 30, 60, 0.06)",
            "card_glow": "rgba(79, 70, 229, 0.07)",
            "app_grad_a": "rgba(79, 70, 229, 0.05)",
            "app_grad_b": "rgba(16, 185, 129, 0.04)",
            "app_grad_c": "#f8faff",
            "sidebar_grad_a": "#fdfeff",
            "sidebar_grad_b": "#f3f7fc",
            "hero_grad_a": "rgba(255,255,255,0.97)",
            "hero_grad_b": "rgba(247,250,255,0.97)",
            "hero_grad_c": "rgba(240,246,254,0.98)",
            "panel_grad_a": "rgba(255,255,255,1)",
            "panel_grad_b": "rgba(248,251,255,1)",
            "metric_grad_a": "rgba(255,255,255,1)",
            "metric_grad_b": "rgba(246,250,255,1)",
            "input_bg": "#ffffff",
            "table_head": "#f4f8fc",
            "table_body": "#ffffff",
            "table_hover": "#f7faff",
            "plot_bg": "rgba(255,255,255,0)",
            "plot_text": "#0f172a",
            "plot_grid": "rgba(15, 30, 60, 0.08)",
            "plot_line": "rgba(15, 30, 60, 0.13)",
            "plot_tick": "#64748b",
        }
    return {
        "bg": "#06101e",
        "bg_soft": "#091524",
        "surface": "#0d1829",
        "surface_2": "#101f35",
        "surface_3": "#13253f",
        "line": "rgba(99, 124, 172, 0.17)",
        "line_strong": "rgba(120, 146, 194, 0.28)",
        "text": "#e8eeff",
        "muted": "#8da0c0",
        "muted_2": "#637290",
        "accent": "#6366f1",
        "accent_2": "#a5b4fc",
        "accent_soft": "rgba(99, 102, 241, 0.12)",
        "good": "#34d399",
        "warn": "#fb923c",
        "bad": "#f87171",
        "shadow": "rgba(0, 0, 0, 0.30)",
        "card_glow": "rgba(99, 102, 241, 0.16)",
        "app_grad_a": "rgba(99, 102, 241, 0.11)",
        "app_grad_b": "rgba(56, 114, 255, 0.07)",
        "app_grad_c": "#070e1b",
        "sidebar_grad_a": "#060d19",
        "sidebar_grad_b": "#0a1422",
        "hero_grad_a": "rgba(17, 32, 60, 0.97)",
        "hero_grad_b": "rgba(10, 22, 44, 0.97)",
        "hero_grad_c": "rgba(6, 14, 30, 0.99)",
        "panel_grad_a": "rgba(13, 23, 42, 0.97)",
        "panel_grad_b": "rgba(8, 16, 32, 0.99)",
        "metric_grad_a": "rgba(12, 22, 40, 0.97)",
        "metric_grad_b": "rgba(7, 14, 28, 0.99)",
        "input_bg": "#0c1826",
        "table_head": "#0d1c2e",
        "table_body": "#0a1726",
        "table_hover": "#0f1f34",
        "plot_bg": "rgba(0,0,0,0)",
        "plot_text": "#dde5ff",
        "plot_grid": "rgba(85, 110, 160, 0.11)",
        "plot_line": "rgba(85, 110, 160, 0.17)",
        "plot_tick": "#7a8fb0",
    }
