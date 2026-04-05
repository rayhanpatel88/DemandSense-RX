"""Shared Streamlit UI helpers for DemandSense-RX."""

from __future__ import annotations

import base64
from pathlib import Path
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

PLOTLY_THEME = {
    "paper_bgcolor": "#07111f",
    "plot_bgcolor": "#07111f",
    "font": {"family": "IBM Plex Sans, sans-serif", "color": "#dfe6ff"},
    "legend": {"font": {"color": "#9ca8c7"}},
}


@st.cache_resource(show_spinner="Running DemandSense-RX pipeline...")
def get_pipeline_data() -> dict:
    from src.pipeline import run_pipeline
    from src.utils.config import load_config

    return run_pipeline(load_config())


def apply_page_config(page_title: str) -> None:
    st.set_page_config(page_title=page_title, page_icon="■", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');
        :root{
            --bg:#060d1b;
            --bg-soft:#081224;
            --surface:#0b1528;
            --surface-2:#0f1b31;
            --surface-3:#13213d;
            --line:rgba(79,104,153,0.24);
            --line-strong:rgba(124,145,197,0.38);
            --text:#e6ecff;
            --muted:#93a2c7;
            --muted-2:#637396;
            --accent:#8f84ff;
            --accent-2:#c7c2ff;
            --accent-soft:rgba(143,132,255,0.12);
            --good:#87d2bc;
            --warn:#ffc57f;
            --bad:#ff8d9f;
        }
        html, body, [class*="css"] { font-family:'IBM Plex Sans', sans-serif; }
        .stApp {
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(143,132,255,0.14), transparent 30%),
                radial-gradient(circle at top right, rgba(90,129,255,0.10), transparent 28%),
                linear-gradient(180deg, #07101e 0%, #040913 100%);
        }
        .block-container { max-width: 1500px; padding-top: 1.0rem; padding-bottom: 2rem; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #070d19 0%, #081120 100%);
            border-right: 1px solid rgba(74,96,141,0.24);
        }
        [data-testid="stSidebar"] * { color: var(--text) !important; }
        [data-testid="stSidebar"] .stButton button {
            background: linear-gradient(135deg, #5a44ff 0%, #9f97ff 100%);
            color: #f7f6ff !important;
            border: none;
            border-radius: 6px;
            font-size: 0.69rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            min-height: 2.6rem;
            box-shadow: 0 14px 32px rgba(90,68,255,0.24);
        }
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
            border-left: 2px solid transparent;
            border-radius: 0;
            padding: 0.62rem 0.75rem;
            background: transparent;
            transition: background-color 0.18s ease, border-color 0.18s ease;
        }
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {
            background: rgba(255,255,255,0.04);
            border-left-color: rgba(143,132,255,0.6);
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
        .page-shell {
            padding: 1.1rem 0 1.5rem 0;
            margin-bottom: 1.3rem;
            border-bottom: 1px solid rgba(82,103,149,0.18);
        }
        .page-header-grid {
            display:grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 1rem;
            align-items:end;
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
            font-size: 3rem;
            line-height: 0.98;
            font-weight: 700;
            margin: 0 0 0.45rem 0;
            text-transform: uppercase;
        }
        .page-subtitle {
            max-width: 62rem;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.55;
        }
        .page-meta {
            display:flex;
            flex-direction:column;
            align-items:flex-end;
            gap:0.4rem;
            text-align:right;
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
                linear-gradient(180deg, rgba(18,30,55,0.92) 0%, rgba(10,19,35,0.96) 100%);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.05rem 1.1rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.22);
        }
        .panel-tight { padding: 0.82rem 0.95rem; }
        .panel-hero {
            background:
                linear-gradient(135deg, rgba(29,46,83,0.96) 0%, rgba(16,26,48,0.96) 65%, rgba(11,18,31,0.98) 100%);
            border-left: 3px solid var(--accent);
        }
        .metric-panel {
            position:relative;
            overflow:hidden;
            background: linear-gradient(180deg, rgba(15,27,49,0.96) 0%, rgba(10,18,33,0.98) 100%);
            border: 1px solid var(--line);
            border-radius: 6px;
            padding: 1rem 1rem 0.95rem 1.1rem;
            min-height: 122px;
        }
        .metric-panel::before {
            content:"";
            position:absolute;
            left:0;
            top:0;
            bottom:0;
            width:3px;
            background: linear-gradient(180deg, var(--accent) 0%, rgba(143,132,255,0.2) 100%);
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.66rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
        }
        .metric-value {
            font-family:'Space Grotesk', sans-serif;
            font-size: 2rem;
            line-height: 1.04;
            font-weight: 700;
            margin-top: 0.38rem;
            color: #f4f6ff;
        }
        .metric-detail {
            margin-top: 0.55rem;
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.45;
        }
        .note {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.58;
        }
        .tag {
            display:inline-flex;
            align-items:center;
            padding:0.18rem 0.46rem;
            margin-right:0.42rem;
            border:1px solid rgba(123,145,197,0.22);
            border-radius:4px;
            background: rgba(255,255,255,0.025);
            color:#e4e9fb;
            font-family:'IBM Plex Mono', monospace;
            font-size:0.68rem;
            letter-spacing:0.04em;
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
            box-shadow:0 0 14px rgba(143,132,255,0.7);
        }
        .brand-lockup {
            display:flex;
            align-items:center;
            gap:0.8rem;
            margin-bottom:0.7rem;
        }
        .brand-mark {
            width:34px;
            height:34px;
            border-radius:6px;
            display:flex;
            align-items:center;
            justify-content:center;
            overflow:hidden;
            box-shadow:0 18px 34px rgba(90,68,255,0.24);
        }
        .brand-mark img { width:100%; height:100%; display:block; }
        .brand-name {
            font-family:'Space Grotesk', sans-serif;
            font-size:1.0rem;
            line-height:1;
            letter-spacing:-0.03em;
            font-weight:700;
            text-transform:uppercase;
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
            background:linear-gradient(180deg, rgba(14,23,41,0.96) 0%, rgba(10,17,31,0.98) 100%);
            border:1px solid rgba(76,97,139,0.24);
            border-radius:6px;
            padding:0.75rem 0.85rem;
            margin:0.35rem 0 0.75rem 0;
        }
        .sidebar-kpi strong {
            display:block;
            color:#f4f6ff;
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
            padding:0.24rem 0.52rem;
            background:rgba(90,68,255,0.24);
            border:1px solid rgba(143,132,255,0.32);
            border-radius:4px;
            color:#e9e7ff;
            font-size:0.62rem;
            font-weight:700;
            letter-spacing:0.16em;
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
            background:rgba(255,255,255,0.02);
            border:1px solid rgba(79,104,153,0.18);
            border-radius:6px;
            padding:0.72rem 0.78rem;
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
            border:1px solid rgba(79,104,153,0.18);
            border-radius:6px;
            overflow:hidden;
            background:#09111f;
        }
        div[data-testid="stDataFrame"] [role="table"] { color:#e3e9fb; }
        div[data-testid="stDataFrame"] thead tr th {
            background:#0d1830 !important;
            color:#94a5ca !important;
            font-size:0.67rem !important;
            text-transform:uppercase !important;
            letter-spacing:0.12em !important;
            border-bottom:1px solid rgba(79,104,153,0.16) !important;
        }
        div[data-testid="stDataFrame"] tbody tr td {
            background:#09111f !important;
            border-bottom:1px solid rgba(79,104,153,0.08) !important;
            font-size:0.77rem !important;
        }
        div[data-testid="stDataFrame"] tbody tr:hover td { background:#0b1528 !important; }
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div,
        .stTextInput input,
        .stNumberInput input {
            background:#0b1528;
            border:1px solid rgba(79,104,153,0.22);
            color:#e8edff;
            border-radius:6px;
        }
        .stSlider [data-baseweb="slider"] { padding-top:0.25rem; }
        .stSlider [role="slider"] { background: var(--accent); }
        .stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] { background: rgba(143,132,255,0.18); }
        .stSegmentedControl [data-baseweb="button-group"] {
            background:#0a1426;
            border:1px solid rgba(79,104,153,0.22);
            border-radius:6px;
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
            background:linear-gradient(135deg, rgba(90,68,255,0.95) 0%, rgba(159,151,255,0.95) 100%) !important;
            color:#f7f6ff !important;
        }
        .stToggle [data-baseweb="checkbox"] > label { color: var(--muted) !important; }
        .stCaption { color: var(--muted) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
                    <div class="brand-sub">Operational Intelligence</div>
                </div>
            </div>
            <div class="security-pill">Security Class: Tier 4</div>
            """,
            unsafe_allow_html=True,
        )
        if data is not None:
            future_df = data["future_df"]
            inventory_df = data["inventory_df"]
            st.markdown(
                f"""
                <div class="sidebar-kpi">
                    <span>Forecast Volume</span>
                    <strong>{int(future_df['forecast'].sum()):,} units</strong>
                </div>
                <div class="sidebar-kpi">
                    <span>Action Queue</span>
                    <strong>{int(inventory_df['reorder_needed'].sum())} SKUs flagged</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            """
            <div class="status-strip" style="margin-top:0.25rem; margin-bottom:0.85rem;">
                <div class="status-item"><span class="status-dot"></span>System Health</div>
                <div class="status-item">100% Operational</div>
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
        st.button("Run New Scenario", use_container_width=True)
        st.caption("Authenticated session · Principal Architect")


def render_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-shell">
            <div class="page-header-grid">
                <div>
                    <div class="page-kicker">{kicker}</div>
                    <div class="page-title">{title}</div>
                    <div class="page-subtitle">{subtitle}</div>
                    <div class="status-strip">
                        <div class="status-item"><span class="status-dot"></span>Live Stream</div>
                        <div class="status-item">Cluster RX-992</div>
                        <div class="status-item">Latency 12ms</div>
                    </div>
                </div>
                <div class="page-meta">
                    <div class="page-meta-top">Systems Intelligence Status</div>
                    <div class="page-meta-bottom">Authenticated // Live // Institutional Build</div>
                </div>
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
        "</div>"
    )


def style_plotly(fig, height: int = 320):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), **PLOTLY_THEME)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(63,84,128,0.15)", zeroline=False, linecolor="rgba(104,122,165,0.18)", tickfont={"color": "#8391b2"})
    fig.update_yaxes(showgrid=True, gridcolor="rgba(63,84,128,0.15)", zeroline=False, linecolor="rgba(104,122,165,0.18)", tickfont={"color": "#8391b2"})
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
    st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def _logo_data_uri() -> str:
    path = Path(__file__).resolve().parents[2] / "assets" / "demandsense_mark.svg"
    if not path.exists():
        fallback = (
            "<svg xmlns='http://www.w3.org/2000/svg' width='96' height='96' viewBox='0 0 96 96'>"
            "<rect x='6' y='6' width='84' height='84' rx='18' fill='url(#g)'/>"
            "<path d='M28 67V29H45.5C58.2 29 66 36.1 66 48C66 60.8 57.5 67 44.8 67H28Z' fill='#07111F'/>"
            "<defs><linearGradient id='g' x1='18' y1='14' x2='78' y2='82'><stop stop-color='#D9DCFF'/><stop offset='1' stop-color='#5042F1'/></linearGradient></defs></svg>"
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
