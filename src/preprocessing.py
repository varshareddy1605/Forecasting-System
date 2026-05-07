"""
preprocessing.py
----------------
Loads the Excel dataset, cleans it, and resamples to weekly frequency (W-SAT).

Steps:
  1. Read Excel file
  2. Parse Date column to datetime
  3. Sort by State and Date
  4. Group by State → resample to weekly (W-SAT) → aggregate Total with sum
  5. Forward-fill any remaining missing values

Output: dict[state_name -> pd.DataFrame] with columns [Date, Total]
"""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Default path relative to project root
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Forecasting Case- Study.xlsx"


def load_and_preprocess(data_path: Path = DEFAULT_DATA_PATH) -> dict[str, pd.DataFrame]:
    """
    Load the Excel file and return a per-state dictionary of weekly time series.

    Parameters
    ----------
    data_path : Path
        Path to the Excel file.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are state names; values are DataFrames indexed by Date (W-SAT freq)
        with a single column 'Total'.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading data from: {data_path}")
    df = pd.read_excel(data_path)
    logger.info(f"Raw data shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    # ── Normalise column names ──────────────────────────────────────────────
    df.columns = df.columns.str.strip()

    # Identify the relevant columns (case-insensitive search)
    col_map = {c.lower(): c for c in df.columns}
    state_col = col_map.get("state")
    date_col  = col_map.get("date")
    total_col = col_map.get("total")

    if not all([state_col, date_col, total_col]):
        raise ValueError(
            f"Expected columns 'State', 'Date', 'Total'. Found: {df.columns.tolist()}"
        )

    df = df[[state_col, date_col, total_col]].copy()
    df.columns = ["State", "Date", "Total"]

    # ── Parse dates ─────────────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"])

    # ── Drop rows where Total is null before resampling ─────────────────────
    df.dropna(subset=["Total"], inplace=True)

    # ── Sort ─────────────────────────────────────────────────────────────────
    df.sort_values(["State", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Per-state resample to W-SAT ─────────────────────────────────────────
    state_data: dict[str, pd.DataFrame] = {}

    for state, group in df.groupby("State"):
        ts = group.set_index("Date")[["Total"]]
        # Resample to weekly Saturday frequency, summing sales
        ts_weekly = ts.resample("W-SAT").sum()
        # Forward-fill any gaps created by resampling
        ts_weekly["Total"] = ts_weekly["Total"].ffill()
        ts_weekly.index.name = "Date"
        state_data[state] = ts_weekly
        logger.debug(f"  {state}: {len(ts_weekly)} weekly records "
                     f"({ts_weekly.index.min().date()} → {ts_weekly.index.max().date()})")

    logger.info(f"Preprocessing complete. States found: {len(state_data)}")
    return state_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = load_and_preprocess()
    for state, df in data.items():
        print(f"{state}: {len(df)} rows | {df.index.min().date()} → {df.index.max().date()}")
