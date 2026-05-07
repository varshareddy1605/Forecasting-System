"""
feature_engineering.py
-----------------------
Creates all time-series features needed for ML models (XGBoost, LSTM).

Feature Groups
==============
1. Lag Features (weekly data) — Assignment required: t-1, t-7, t-30
   -----------------------------------------------------------------
   The assignment requires lag features for t-1, t-7, and t-30 days.
   Because the data is resampled to WEEKLY frequency, the mapping is:

       Assignment Requirement | Weekly Implementation
       -----------------------|----------------------
       t-1                    | lag_1  (1 week lag)
       t-7                    | lag_1  (1 week lag — same weekly bin)
       t-30                   | lag_4  (4 week lag)

   Only lag_1 and lag_4 are implemented, strictly following assignment
   requirements. lag_13 is NOT used.

2. Rolling Statistics (required)
   rolling_mean_4  – 4-week rolling mean
   rolling_std_4   – 4-week rolling std

3. Date Features (required: day_of_week, month — extras: quarter, week_of_year)

4. Holiday Feature (required)
   is_holiday – US public holidays (1/0)
"""

import logging
from typing import List

import holidays
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_US_HOLIDAYS = holidays.UnitedStates()


def add_holiday_flag(dates: pd.DatetimeIndex) -> np.ndarray:
    """Return a binary array: 1 if date is a US public holiday, else 0."""
    return np.array([1 if d in _US_HOLIDAYS else 0 for d in dates], dtype=np.int8)


def create_features(df: pd.DataFrame, target_col: str = "Total") -> pd.DataFrame:
    """
    Add lag, rolling, date, and holiday features to a single-state DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex and a `target_col` column.
    target_col : str
        Column name for the sales target.

    Returns
    -------
    pd.DataFrame
        Original dataframe extended with engineered features; rows with NaN
        (due to lag windows) are dropped.
    """
    df = df.copy()
    s = df[target_col]

    # ── Lag features ─────────────────────────────────────────────────────────
    # Assignment requires t-1, t-7, t-30. At weekly granularity:
    #   t-1  → lag_1 (1 week)
    #   t-7  → lag_1 (1 week — same weekly bin)
    #   t-30 → lag_4 (4 weeks)
    df["lag_1"] = s.shift(1)   # covers t-1 and t-7 day requirements
    df["lag_4"] = s.shift(4)   # covers t-30 day requirement

    # ── Rolling features ──────────────────────────────────────────────────────
    df["rolling_mean_4"] = s.shift(1).rolling(window=4).mean()
    df["rolling_std_4"]  = s.shift(1).rolling(window=4).std()

    # ── Date features ─────────────────────────────────────────────────────────
    df["day_of_week"]  = df.index.dayofweek          # 0=Monday … 6=Sunday
    df["month"]        = df.index.month
    df["quarter"]      = df.index.quarter
    df["week_of_year"] = df.index.isocalendar().week.astype(int)

    # ── Holiday flag ─────────────────────────────────────────────────────────
    df["is_holiday"] = add_holiday_flag(df.index)

    # Drop NaN rows created by lags/rolling
    before = len(df)
    df.dropna(inplace=True)
    logger.debug(f"Feature engineering: dropped {before - len(df)} NaN rows "
                 f"(lag/rolling window), {len(df)} rows remaining.")
    return df


def get_feature_columns() -> List[str]:
    """
    Return the ordered list of ML feature columns (excludes target).

    Required by assignment: lag_1, lag_4, rolling_mean_4, rolling_std_4,
    day_of_week, month, is_holiday.
    Additional extras retained: quarter, week_of_year.
    """
    return [
        "lag_1", "lag_4",
        "rolling_mean_4", "rolling_std_4",
        "day_of_week", "month", "quarter", "week_of_year",
        "is_holiday",
    ]


def train_val_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict time-based split — NO shuffling to prevent data leakage.

    Split:
      - Train      : Date < 2023-01-01
      - Validation : 2023-01-01 <= Date < 2023-07-01
      - Test        : 2023-07-01 <= Date <= 2023-12-31
    """
    train = df[df.index < "2023-01-01"]
    val   = df[(df.index >= "2023-01-01") & (df.index < "2023-07-01")]
    test  = df[(df.index >= "2023-07-01") & (df.index <= "2023-12-31")]
    return train, val, test


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
    logging.basicConfig(level=logging.DEBUG)
    from src.preprocessing import load_and_preprocess

    state_data = load_and_preprocess()
    state, raw_df = next(iter(state_data.items()))
    feat_df = create_features(raw_df)
    print(f"\n{state} — feature columns:\n{feat_df.head()}")
    tr, va, te = train_val_test_split(feat_df)
    print(f"Train: {len(tr)}, Val: {len(va)}, Test: {len(te)}")
