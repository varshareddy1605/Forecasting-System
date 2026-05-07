"""
prophet_model.py
----------------
Facebook Prophet model wrapper.
- Renames columns to ds / y (required by Prophet)
- Enables yearly and weekly seasonality
- Adds US country holidays
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def train_prophet(
    train_series: pd.Series,
    val_series: pd.Series,
    test_series: pd.Series,
    state: str,
    artifacts_dir: Path,
) -> Dict:
    """
    Fit Facebook Prophet and evaluate on test set.
    Returns dict with: model_name, metrics, forecast (8 future weeks).
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Install prophet: pip install prophet")

    logger.info(f"[Prophet] Training for state: {state}")

    # Combine train + val for fitting
    fit_series = pd.concat([train_series, val_series])

    # Prophet expects df with 'ds' and 'y' columns
    fit_df = fit_series.reset_index().rename(columns={"Date": "ds", "Total": "y"})
    if "ds" not in fit_df.columns:
        fit_df.columns = ["ds", "y"]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )
    model.add_country_holidays(country_name="US")
    model.fit(fit_df)

    # ── Evaluate on test ─────────────────────────────────────────────────────
    test_df = test_series.reset_index().rename(columns={"Date": "ds", "Total": "y"})
    if "ds" not in test_df.columns:
        test_df.columns = ["ds", "y"]

    test_forecast = model.predict(test_df[["ds"]])
    predicted = test_forecast["yhat"].values
    actual    = test_series.values

    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae  = float(np.mean(np.abs(actual - predicted)))
    mape = _safe_mape(actual, predicted)

    metrics = {"rmse": round(rmse, 4), "mae": round(mae, 4), "mape": round(mape, 4)}
    logger.info(f"[Prophet] {state} metrics: {metrics}")

    # ── 8-week future forecast ───────────────────────────────────────────────
    # Re-fit on all data
    all_series = pd.concat([fit_series, test_series])
    all_df = all_series.reset_index().rename(columns={"Date": "ds", "Total": "y"})
    if "ds" not in all_df.columns:
        all_df.columns = ["ds", "y"]

    final_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )
    final_model.add_country_holidays(country_name="US")
    final_model.fit(all_df)

    last_date = all_series.index[-1]
    future_dates = pd.date_range(start=last_date, periods=9, freq="W-SAT")[1:]
    future_df = pd.DataFrame({"ds": future_dates})
    future_forecast = final_model.predict(future_df)

    forecast: List[Dict] = [
        {"date": row["ds"].strftime("%Y-%m-%d"), "sales": round(float(row["yhat"]), 2)}
        for _, row in future_forecast.iterrows()
    ]

    # ── Save model ───────────────────────────────────────────────────────────
    model_path = artifacts_dir / f"{state}_prophet.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    logger.info(f"[Prophet] Model saved to {model_path}")

    return {
        "model_name": "Prophet",
        "model_path": str(model_path),
        "metrics": metrics,
        "forecast": forecast,
    }
