# Time Series Forecasting System

A **production-style, modular** forecasting backend that:

- Trains 4 models per US state (SARIMA, Prophet, XGBoost, LSTM)
- Automatically selects the best model per state based on RMSE
- Serves 8-week sales forecasts via a FastAPI REST API

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup & Installation](#setup--installation)
3. [Training the Models](#training-the-models)
4. [Running the API](#running-the-api)
5. [API Endpoints](#api-endpoints)
6. [Feature Engineering Details](#feature-engineering-details)
7. [Model Architecture](#model-architecture)
8. [Design Decisions](#design-decisions)

---

## Project Structure

```
forecasting_system/
├── data/
│   └── Forecasting Case- Study.xlsx   ← Raw dataset
├── src/
│   ├── preprocessing.py               ← Data loading & weekly resampling
│   ├── feature_engineering.py         ← Lag, rolling, date, holiday features
│   ├── model_selector.py              ← Per-state best-model selection
│   ├── train_all.py                   ← Main training orchestrator
│   └── models/
│       ├── sarima_model.py
│       ├── prophet_model.py
│       ├── xgboost_model.py
│       └── lstm_model.py
├── api/
│   └── app.py                         ← FastAPI REST API
├── artifacts/
│   ├── models/                        ← Saved model files (.pkl / .keras)
│   ├── forecasts/                     ← Per-state JSON forecasts
│   └── results/                       ← Consolidated evaluation results
├── requirements.txt
└── README.md
```

---

## Setup & Installation

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt
```

> **Note:** If you do not have a GPU, replace `tensorflow` with `tensorflow-cpu` in requirements.txt.

---

## Training the Models

```bash
# Train all 4 models for every state (may take 20–60 min)
python src/train_all.py

# Skip LSTM for a quick test run (~5–15 min)
python src/train_all.py --skip-lstm

# Skip SARIMA as well for the fastest run
python src/train_all.py --skip-lstm --skip-sarima
```

Training logs are written to `training.log` and printed to stdout.  
Trained models are saved to `artifacts/models/`.  
Per-state forecasts are saved to `artifacts/forecasts/<State>.json`.

---

## Running the API

The API requires training to be completed first so that forecast JSON files exist.

```bash
# From the project root
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Open the interactive Swagger UI at: **http://localhost:8000/docs**

---

## API Endpoints

| Method | Endpoint             | Description                                    |
|--------|----------------------|------------------------------------------------|
| GET    | `/`                  | Health check                                   |
| GET    | `/states`            | List all states with available forecasts       |
| GET    | `/forecast/{state}`  | 8-week forecast + metrics for a given state    |
| GET    | `/forecast/all`      | Forecasts for all states                       |
| GET    | `/models/{state}`    | Full model comparison metrics for a state      |
| POST   | `/retrain`           | Trigger background retraining of all models    |

### Example Response — `GET /forecast/California`

```json
{
  "state": "California",
  "best_model": "XGBoost",
  "forecast": [
    {"date": "2024-01-06", "sales": 142500.75},
    {"date": "2024-01-13", "sales": 148320.10}
  ],
  "metrics": {
    "rmse": 12043.22,
    "mae": 9821.45,
    "mape": 7.34
  }
}
```

---

## Feature Engineering Details

### Lag Feature Mapping

> **Important:** Because the raw data is resampled to **weekly frequency (W-SAT)**, the
> assignment's day-level lag requirements map to weekly lags as follows:

| Assignment Requirement | Weekly Implementation | Rationale |
|---|---|---|
| t-1 day | `lag_1` (1 week lag) | Finest granularity available at weekly frequency |
| t-7 days | `lag_1` (1 week lag) | Same weekly bin as t-1 |
| t-30 days | `lag_4` (4 week lag) | 4 weeks ≈ 30 days |

Only `lag_1` and `lag_4` are implemented, strictly following assignment requirements. `lag_13` is **not used**.

### Full Feature List

| Feature          | Required? | Description                                   |
|------------------|-----------|-----------------------------------------------|
| `lag_1`          | ✅ Yes    | Sales 1 week ago (covers t-1 and t-7 days)    |
| `lag_4`          | ✅ Yes    | Sales 4 weeks ago (covers t-30 days)          |
| `rolling_mean_4` | ✅ Yes    | 4-week rolling average of sales               |
| `rolling_std_4`  | ✅ Yes    | 4-week rolling standard deviation             |
| `day_of_week`    | ✅ Yes    | Day of week (always Sat=5 for W-SAT data)     |
| `month`          | ✅ Yes    | Month number (1–12)                           |
| `is_holiday`     | ✅ Yes    | 1 if the date is a US public holiday, else 0  |
| `quarter`        | Extra     | Quarter (1–4) — additional context feature    |
| `week_of_year`   | Extra     | ISO week number — additional context feature  |

> **Note:** `day_of_week` is always 5 (Saturday) for W-SAT resampled data. It is
> retained as a mandatory feature per requirements; it contributes no variance
> but preserves the feature contract for production consistency.

---

## Model Architecture

### Train / Validation / Test Split (No Data Leakage)

| Split      | Date Range              | Purpose                    |
|------------|-------------------------|----------------------------|
| Train      | All data before 2023    | Model fitting              |
| Validation | Jan 2023 – Jun 2023     | Hyperparameter tuning      |
| Test       | Jul 2023 – Dec 2023     | Final evaluation (RMSE)    |

Data is **never shuffled**. Splits are strictly time-ordered.

### SARIMA
- Uses `pmdarima.auto_arima` with `seasonal=True`, `m=52` (weekly seasonality)
- AIC-based order selection
- Stepwise search for speed

### Facebook Prophet
- `yearly_seasonality=True`, `weekly_seasonality=True`
- Multiplicative seasonality mode
- US country holidays added

### XGBoost
- Features: all 10 engineered features listed above
- **Walk-forward forecasting**: predict one week → append → recompute features → repeat × 8
- `TimeSeriesSplit(n_splits=5)` for cross-validation
- 300 estimators, learning rate 0.05

### LSTM
- Sequence length: **12 timesteps** (12 weeks of history → predict week 13)
- Architecture: `LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(1)`
- Optimizer: Adam (lr=0.001), loss: MSE
- Early stopping: patience=10 on validation loss
- **Epochs: 50** ← *reduced for demo purposes; increase to 200+ for production*
- **Walk-forward forecasting**: same iterative approach as XGBoost

---

## Design Decisions

### Walk-Forward Forecasting
Both XGBoost and LSTM use an **iterative walk-forward** approach for the 8-step
forecast horizon:
1. Use the current history window to predict the next week
2. Append the prediction to the history
3. Recompute lag/rolling features from the updated history
4. Repeat until all 8 weeks are predicted

This avoids the need for a "direct multi-output" model and better reflects
real-world deployment where predictions are made one step at a time.

### Per-State Model Selection
Each state gets its own independently trained set of models. The best model
per state is selected based on the **lowest RMSE on the test set** (Jul–Dec 2023).
This means different states may use different model types.

### No Data Leakage
- All feature computation (lags, rolling) uses `.shift(1)` — no look-ahead
- Train/Val/Test splits are strictly time-ordered
- Scalers (MinMaxScaler for LSTM) are fit only on training data

---

## Evaluation Metrics

| Metric | Description                              |
|--------|------------------------------------------|
| RMSE   | Root Mean Squared Error (primary metric) |
| MAE    | Mean Absolute Error                      |
| MAPE   | Mean Absolute Percentage Error (%)       |
