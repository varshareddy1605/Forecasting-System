"""
sarima_model.py
---------------
SARIMA model wrapper using pmdarima (auto_arima).
Seasonal period = 52 weeks.
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


def train_sarima(
    train_series: pd.Series,
    val_series: pd.Series,
    test_series: pd.Series,
    state: str,
    artifacts_dir: Path,
) -> Dict:
    """
    Fit SARIMA via auto_arima and evaluate on test set.

    Returns dict with: model, metrics, forecast (8 future weeks)
    """
    try:
        import pmdarima as pm
    except ImportError:
        raise ImportError("Install pmdarima: pip install pmdarima")

    logger.info(f"[SARIMA] Training for state: {state}")

    # Combine train + val for final fit
    fit_series = pd.concat([train_series, val_series])

    model = pm.auto_arima(
        fit_series,
        seasonal=True,
        m=52,
        stepwise=True,
        approximation=True,
        error_action="ignore",
        suppress_warnings=True,
        max_p=1, max_q=1,
        max_P=1, max_Q=1,
        max_d=1, max_D=1,
        information_criterion="aic",
        n_jobs=-1,
    )
    logger.info(f"[SARIMA] {state} — best order: {model.order} seasonal: {model.seasonal_order}")

    # ── Evaluate on test ─────────────────────────────────────────────────────
    n_test = len(test_series)
    test_preds = model.predict(n_periods=n_test)

    actual    = test_series.values
    predicted = np.array(test_preds)
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae  = float(np.mean(np.abs(actual - predicted)))
    mape = _safe_mape(actual, predicted)

    metrics = {"rmse": round(rmse, 4), "mae": round(mae, 4), "mape": round(mape, 4)}
    logger.info(f"[SARIMA] {state} metrics: {metrics}")

    # ── 8-week future forecast ───────────────────────────────────────────────
    # Re-fit on all available data (train + val + test)
    all_series = pd.concat([fit_series, test_series])
    model.update(test_series)
    future_preds = model.predict(n_periods=8)

    last_date = all_series.index[-1]
    future_dates = pd.date_range(start=last_date, periods=9, freq="W-SAT")[1:]
    forecast: List[Dict] = [
        {"date": d.strftime("%Y-%m-%d"), "sales": round(float(v), 2)}
        for d, v in zip(future_dates, future_preds)
    ]

    # ── Save model ───────────────────────────────────────────────────────────
    model_path = artifacts_dir / f"{state}_sarima.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"[SARIMA] Model saved to {model_path}")

    return {
        "model_name": "SARIMA",
        "model_path": str(model_path),
        "metrics": metrics,
        "forecast": forecast,
    }
