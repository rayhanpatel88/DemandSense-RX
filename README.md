# DemandSense-RX

**Autonomous Demand Forecasting and Robotic Fulfilment Intelligence System**

A production-grade machine learning system that forecasts SKU-level demand, quantifies uncertainty, generates inventory decisions, and simulates warehouse robotic fulfilment — all in an interactive Streamlit dashboard.

---

## Architecture

```
DemandSense-RX/
├── streamlit_app.py          # Entry point (Executive Overview page)
├── pages/                    # Streamlit multi-page app
│   ├── 1_Forecast_Explorer.py
│   ├── 2_Inventory_Decisions.py
│   ├── 3_Robotics_Simulation.py
│   ├── 4_Explainability.py
│   ├── 5_Backtesting.py
│   └── 6_Scenario_Simulator.py
├── src/
│   ├── pipeline.py           # Full pipeline orchestrator
│   ├── data/                 # Synthetic data generation & loading
│   ├── features/             # Feature engineering
│   ├── models/               # LightGBM + baseline forecasters
│   ├── evaluation/           # Metrics & rolling backtesting
│   ├── explainability/       # SHAP-based model explanations
│   ├── recommendations/      # Inventory intelligence (safety stock, ROP)
│   ├── simulation/           # Warehouse robotics (A*, multi-agent)
│   └── utils/                # Config loader, logger
├── configs/
│   └── config.yaml           # All system parameters
├── data/raw/                 # Auto-generated synthetic data (cached)
├── models/                   # Saved model artefacts
└── tests/                    # Unit tests
```

### Data Flow

```
Synthetic Data → Feature Engineering → LightGBM Training
       ↓
Forecast Generation (point + 80% intervals)
       ↓
┌─────────────────┬────────────────────┬─────────────────┐
│ Inventory Engine│  Backtesting Engine│  SHAP Explainer │
│ (safety stock,  │  (4-fold rolling,  │  (global + local│
│  reorder point) │   3 models)        │   explanations) │
└─────────────────┴────────────────────┴─────────────────┘
       ↓
Warehouse Robotics Simulation (A* pathfinding, multi-agent)
       ↓
Streamlit Dashboard (7 pages)
```

---

## Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/your-username/DemandSense-RX.git
cd DemandSense-RX
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run streamlit_app.py
```

The first run will:
1. Generate synthetic demand data for 20 SKUs (saved to `data/raw/`)
2. Train LightGBM and baseline models
3. Run rolling backtesting (4 folds)
4. Compute SHAP values
5. Open the dashboard in your browser

Subsequent runs load cached data and models for fast startup.

---

## Features

### 1. Demand Forecasting (LightGBM)

- Global multi-SKU LightGBM model with SKU-level encoding
- Features: lag (1, 7, 14, 28 days), rolling mean/std (7, 14, 28 days), calendar, price, promotions
- **80% prediction intervals** via quantile regression (q=0.1 and q=0.9 models)
- **Reliability score** per prediction based on interval width
- Confidence flags: `high`, `medium`, `low`

### 2. Baseline Models

- **Moving Average**: Rolling 7-day window mean per SKU
- **Seasonal Naive**: Uses same day from previous week (lag_7)

### 3. Backtesting Engine

- Rolling-origin evaluation with 4 folds × 30-day test windows
- Metrics: **MAE**, **RMSE**, **MAPE**, **WAPE**
- Per-model and per-SKU breakdown

### 4. Inventory Intelligence

Based on classical inventory theory with configurable service levels:

```
Safety Stock = z × σ_demand × √(lead_time)
Reorder Point = mean_demand × lead_time + safety_stock
Days to Stockout = current_stock / mean_daily_demand
```

Risk flags: `critical` (<1× lead time), `high` (<2×), `medium` (<3×), `low`

### 5. SHAP Explainability

- **Global**: Mean |SHAP| value per feature (top drivers of demand)
- **Distribution**: Beeswarm-style plot showing feature impact across predictions
- **Local**: Single-prediction waterfall chart with feature values

### 6. Warehouse Robotics Simulation

**Environment**: 20×15 grid with shelf rows, aisles, and packing stations

**Algorithm**: A* pathfinding with:
- Manhattan distance heuristic
- Obstacle avoidance (shelves are walls)
- Dynamic collision avoidance (occupied cell tracking)

**Robot lifecycle**:
```
IDLE → MOVING_TO_SHELF → PICKING (2 steps) → MOVING_TO_PACKING → DELIVERING → IDLE
```

**Metrics**: fulfilment rate, avg pick time, robot utilisation %, congestion heatmap, delayed orders

### 7. Scenario Simulator

Adjustable levers:
- Demand multiplier (0.5× – 3.0×)
- Price change (with -0.5 elasticity approximation)
- Lead time
- Service level
- Number of robots

---

## Configuration

All parameters live in `configs/config.yaml`:

```yaml
data:
  n_skus: 20
  start_date: "2022-01-01"
  end_date: "2024-01-01"

forecasting:
  default_horizon: 30
  lag_features: [1, 7, 14, 28]
  rolling_windows: [7, 14, 28]

models:
  lgbm:
    n_estimators: 300
    learning_rate: 0.05

inventory:
  default_lead_time_days: 7
  default_service_level: 0.95

simulation:
  grid_width: 20
  grid_height: 15
  n_robots: 3
  time_steps: 100
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests cover:
- Feature engineering (column creation, lag correctness, row count preservation)
- Inventory logic (safety stock, reorder point, risk flags, lead time sensitivity)
- A* pathfinding (obstacle avoidance, path continuity, no-path detection)

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| Executive Overview | KPI cards, demand summary, stockout risk distribution |
| Forecast Explorer | Per-SKU time series with confidence intervals |
| Inventory Decisions | Reorder table, safety stock, stockout timeline |
| Robotics Simulation | Animated grid, congestion heatmap, robot metrics |
| Explainability | SHAP importance, distribution plot, local explanation |
| Backtesting | Model comparison, error trends, actual vs predicted |
| Scenario Simulator | What-if analysis with dynamic output updates |

---

## Technology Stack

| Component | Library |
|-----------|---------|
| Forecasting | LightGBM (quantile regression) |
| Feature Engineering | pandas, numpy |
| Backtesting | Custom rolling-origin engine |
| Explainability | SHAP (TreeExplainer) |
| Inventory | scipy (Normal distribution) |
| Pathfinding | Custom A* implementation |
| Visualisation | Plotly |
| App | Streamlit |
| Config | PyYAML |
