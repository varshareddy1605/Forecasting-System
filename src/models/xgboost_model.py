"""
xgboost_model.py
----------------
XGBoost Regressor with lag + rolling + date features.
Uses walk-forward forecasting for multi-step (8-week) prediction:
  1. Predict one step ahead
  2. Append prediction to history
  3. Recompute features
  4. Repeat for 8 iterations
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "lag_1", "lag_4",              # required by assignment (t-1, t-7 → lag_1; t-30 → lag_4)
    "rolling_mean_4", "rolling_std_4",
    "day_of_week", "month", "quarter", "week_of_year",
    "is_holiday",
]


def _safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _recompute_features(history: pd.Series, target_date: pd.Timestamp) -> pd.Series:
    """
    Given a history series and a target date, compute a single row of features.
    Used during walk-forward forecasting.
    """
    import holidays as hol_lib
    us_holidays = hol_lib.UnitedStates()

    def lag(n):
        if len(history) >= n:
            return history.iloc[-n]
        return np.nan

    def rolling_mean(window):
        if len(history) >= 1:
            return history.iloc[-min(len(history), window):].mean()
        return np.nan

    def rolling_std(window):
        if len(history) >= 2:
            return history.iloc[-min(len(history), window):].std()
        return np.nan

    return pd.Series({
        "lag_1":          lag(1),
        "lag_4":          lag(4),
        "lag_13":         lag(13),
        "rolling_mean_4": rolling_mean(4),
        "rolling_std_4":  rolling_std(4),
        "day_of_week":    target_date.dayofweek,
        "month":          target_date.month,
        "quarter":        target_date.quarter,
        "week_of_year":   target_date.isocalendar().week,
        "is_holiday":     int(target_date in us_holidays),
    })


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    state: str,
    artifacts_dir: Path,
) -> Dict:
    """
    Train XGBoost regressor with time-series cross-validation.
    Evaluate on test set using walk-forward forecasting.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Feature-engineered DataFrames with FEATURE_COLS + 'Total'.
    state : str
    artifacts_dir : Path

    Returns
    -------
    dict with model_name, metrics, forecast
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("Install xgboost: pip install xgboost")

    logger.info(f"[XGBoost] Training for state: {state}")

    # Combine train + val for model fitting
    fit_df = pd.concat([train_df, val_df])
    X_fit = fit_df[FEATURE_COLS]
    y_fit = fit_df["Total"]

    tscv = TimeSeriesSplit(n_splits=5)
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_fit, y_fit)],
        verbose=False,
    )

    # ── Evaluate on test using walk-forward ─────────────────────────────────
    history = fit_df["Total"].copy()
    test_preds = []

    for i, (date, row) in enumerate(test_df.iterrows()):
        feat = _recompute_features(history, date).to_frame().T
        pred = float(model.predict(feat[FEATURE_COLS])[0])
        test_preds.append(pred)
        history = pd.concat([history, pd.Series([pred], index=[date])])

    actual    = test_df["Total"].values
    predicted = np.array(test_preds)

    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae  = float(np.mean(np.abs(actual - predicted)))
    mape = _safe_mape(actual, predicted)

    metrics = {"rmse": round(rmse, 4), "mae": round(mae, 4), "mape": round(mape, 4)}
    logger.info(f"[XGBoost] {state} metrics: {metrics}")

    # ── 8-week walk-forward forecast ─────────────────────────────────────────
    # Refit on all data
    all_df = pd.concat([fit_df, test_df])
    X_all  = all_df[FEATURE_COLS]
    y_all  = all_df["Total"]
    model.fit(X_all, y_all, verbose=False)

    history = all_df["Total"].copy()
    last_date = all_df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=9, freq="W-SAT")[1:]

    forecast: List[Dict] = []
    for fd in future_dates:
        feat = _recompute_features(history, fd).to_frame().T
        pred = float(model.predict(feat[FEATURE_COLS])[0])
        history = pd.concat([history, pd.Series([pred], index=[fd])])
        forecast.append({"date": fd.strftime("%Y-%m-%d"), "sales": round(pred, 2)})

    # ── Save model ───────────────────────────────────────────────────────────
    model_path = artifacts_dir / f"{state}_xgboost.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"[XGBoost] Model saved to {model_path}")

    return {
        "model_name": "XGBoost",
        "model_path": str(model_path),
        "metrics": metrics,
        "forecast": forecast,
    }
