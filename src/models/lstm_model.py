"""
lstm_model.py
-------------
LSTM (Deep Learning) model for time-series forecasting.

Architecture:
  LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(1)

Key design choices:
  - Sequence length = 12 weeks
  - MinMaxScaler normalization
  - Walk-forward forecasting for 8-step prediction
  - Early stopping on validation loss
  - Reduced epochs (50) for demo purposes — increase for production
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

SEQ_LEN = 12   # number of past weeks used as input
EPOCHS  = 50   # reduced for demo; increase in production
BATCH   = 16


def _safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _build_sequences(data: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create (X, y) pairs of shape (samples, seq_len, 1) and (samples,)."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


def _build_model(seq_len: int):
    """Build and compile the LSTM architecture."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dropout, Dense
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        raise ImportError("Install tensorflow: pip install tensorflow")

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def train_lstm(
    train_series: pd.Series,
    val_series: pd.Series,
    test_series: pd.Series,
    state: str,
    artifacts_dir: Path,
) -> Dict:
    """
    Train LSTM and evaluate on test set using walk-forward forecasting.

    Parameters
    ----------
    train_series, val_series, test_series : pd.Series
        Raw 'Total' series (before feature engineering).
    state : str
    artifacts_dir : Path

    Returns
    -------
    dict with model_name, metrics, forecast
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        raise ImportError("Install tensorflow: pip install tensorflow")

    logger.info(f"[LSTM] Training for state: {state}")

    # ── Scale ─────────────────────────────────────────────────────────────────
    fit_series = pd.concat([train_series, val_series])
    scaler = MinMaxScaler(feature_range=(0, 1))
    fit_scaled = scaler.fit_transform(fit_series.values.reshape(-1, 1)).flatten()
    val_scaled  = scaler.transform(val_series.values.reshape(-1, 1)).flatten()

    # Build train sequences from fit_series (all except last seq_len for val)
    train_scaled = scaler.transform(train_series.values.reshape(-1, 1)).flatten()
    X_tr, y_tr = _build_sequences(train_scaled, SEQ_LEN)
    X_va, y_va = _build_sequences(
        np.concatenate([train_scaled[-SEQ_LEN:], val_scaled]), SEQ_LEN
    )

    # Reshape for LSTM: (samples, timesteps, features)
    X_tr = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)
    X_va = X_va.reshape(X_va.shape[0], X_va.shape[1], 1)

    model = _build_model(SEQ_LEN)

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=[early_stop],
        verbose=0,
    )
    logger.info(f"[LSTM] {state} — training complete.")

    # ── Walk-forward evaluation on test ──────────────────────────────────────
    # Seed with last SEQ_LEN values from fit_series
    history_scaled = list(fit_scaled[-SEQ_LEN:])
    test_scaled = scaler.transform(test_series.values.reshape(-1, 1)).flatten()

    test_preds_scaled = []
    for actual_val in test_scaled:
        seq = np.array(history_scaled[-SEQ_LEN:]).reshape(1, SEQ_LEN, 1)
        pred = float(model.predict(seq, verbose=0)[0][0])
        test_preds_scaled.append(pred)
        history_scaled.append(actual_val)  # use actual (not pred) during eval

    predicted = scaler.inverse_transform(
        np.array(test_preds_scaled).reshape(-1, 1)
    ).flatten()
    actual = test_series.values

    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae  = float(np.mean(np.abs(actual - predicted)))
    mape = _safe_mape(actual, predicted)

    metrics = {"rmse": round(rmse, 4), "mae": round(mae, 4), "mape": round(mape, 4)}
    logger.info(f"[LSTM] {state} metrics: {metrics}")

    # ── 8-week walk-forward forecast ─────────────────────────────────────────
    # Refit on all data
    all_series = pd.concat([fit_series, test_series])
    all_scaled = scaler.fit_transform(all_series.values.reshape(-1, 1)).flatten()
    X_all, y_all = _build_sequences(all_scaled, SEQ_LEN)
    X_all = X_all.reshape(X_all.shape[0], X_all.shape[1], 1)

    final_model = _build_model(SEQ_LEN)
    final_model.fit(X_all, y_all, epochs=EPOCHS, batch_size=BATCH, verbose=0)

    history_f = list(all_scaled[-SEQ_LEN:])
    last_date = all_series.index[-1]
    future_dates = pd.date_range(start=last_date, periods=9, freq="W-SAT")[1:]

    forecast: List[Dict] = []
    for fd in future_dates:
        seq = np.array(history_f[-SEQ_LEN:]).reshape(1, SEQ_LEN, 1)
        pred_scaled = float(final_model.predict(seq, verbose=0)[0][0])
        pred = float(scaler.inverse_transform([[pred_scaled]])[0][0])
        history_f.append(pred_scaled)
        forecast.append({"date": fd.strftime("%Y-%m-%d"), "sales": round(pred, 2)})

    # ── Save model + scaler ───────────────────────────────────────────────────
    model_path  = artifacts_dir / f"{state}_lstm.pkl"
    scaler_path = artifacts_dir / f"{state}_lstm_scaler.pkl"

    final_model.save(str(model_path).replace(".pkl", ".keras"))
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"[LSTM] Model saved to {model_path}")

    return {
        "model_name": "LSTM",
        "model_path": str(model_path),
        "metrics": metrics,
        "forecast": forecast,
    }
