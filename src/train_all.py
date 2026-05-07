"""
train_all.py
------------
Main training orchestrator. Runs the full pipeline:

  1. Load & preprocess data          (preprocessing.py)
  2. Feature engineering             (feature_engineering.py)
  3. Train/Val/Test split            (no shuffling — strict time order)
  4. Train 4 models per state        (SARIMA, Prophet, XGBoost, LSTM)
  5. Select best model per state     (model_selector.py)
  6. Save forecasts as JSON          (artifacts/forecasts/)
  7. Save evaluation results as JSON (artifacts/results/)

Usage:
    python src/train_all.py
"""

import json
import logging
import sys
import time
import traceback
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import load_and_preprocess
from src.feature_engineering import create_features, train_val_test_split
from src.model_selector import select_best_models_all_states
from src.models.sarima_model import train_sarima
from src.models.prophet_model import train_prophet
from src.models.xgboost_model import train_xgboost
from src.models.lstm_model import train_lstm

# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACTS_MODELS    = ROOT / "artifacts" / "models"
ARTIFACTS_FORECASTS = ROOT / "artifacts" / "forecasts"
ARTIFACTS_RESULTS   = ROOT / "artifacts" / "results"

for d in [ARTIFACTS_MODELS, ARTIFACTS_FORECASTS, ARTIFACTS_RESULTS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "training.log", mode="w"),
    ],
)
logger = logging.getLogger("train_all")


def run_pipeline(skip_lstm: bool = False, skip_sarima: bool = False, limit: int = None) -> None:
    """
    Execute the complete training pipeline.

    Parameters
    ----------
    skip_lstm   : bool  — skip LSTM (useful for quick test runs)
    skip_sarima : bool  — skip SARIMA (slow on large series)
    """
    start = time.time()
    logger.info("=" * 70)
    logger.info("  TIME SERIES FORECASTING SYSTEM — TRAINING PIPELINE")
    logger.info("=" * 70)

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    logger.info("STEP 1 / 5: Loading and preprocessing data ...")
    state_data = load_and_preprocess()
    states = list(state_data.keys())
    if limit is not None:
        states = states[:limit]
        logger.info(f"Limiting to {limit} states for testing.")
    logger.info(f"States to process: {states}")

    # ── Step 2 – 4: Per-state model training ─────────────────────────────────
    all_state_results: dict = {}

    for state in states:
        logger.info("-" * 60)
        logger.info(f"Processing state: {state}")

        raw_df = state_data[state]

        # ── Feature engineering (for ML models) ──────────────────────────────
        feat_df = create_features(raw_df)

        # ── Time-based split ──────────────────────────────────────────────────
        train_feat, val_feat, test_feat = train_val_test_split(feat_df)
        train_raw,  val_raw,  test_raw  = (
            raw_df.loc[train_feat.index]["Total"],
            raw_df.loc[val_feat.index]["Total"],
            raw_df.loc[test_feat.index]["Total"],
        )

        logger.info(
            f"  Split — Train: {len(train_feat)}, Val: {len(val_feat)}, Test: {len(test_feat)}"
        )

        if len(test_feat) == 0:
            logger.warning(f"  No test data for {state} — skipping.")
            continue

        model_results = []

        # ── SARIMA ────────────────────────────────────────────────────────────
        if not skip_sarima:
            try:
                result = train_sarima(
                    train_raw, val_raw, test_raw, state, ARTIFACTS_MODELS
                )
                model_results.append(result)
            except Exception as e:
                logger.error(f"  [SARIMA] Failed for {state}: {e}")
                logger.debug(traceback.format_exc())
        else:
            logger.info("  [SARIMA] Skipped.")

        # ── Prophet ───────────────────────────────────────────────────────────
        try:
            result = train_prophet(
                train_raw, val_raw, test_raw, state, ARTIFACTS_MODELS
            )
            model_results.append(result)
        except Exception as e:
            logger.error(f"  [Prophet] Failed for {state}: {e}")
            logger.debug(traceback.format_exc())

        # ── XGBoost ───────────────────────────────────────────────────────────
        try:
            result = train_xgboost(
                train_feat, val_feat, test_feat, state, ARTIFACTS_MODELS
            )
            model_results.append(result)
        except Exception as e:
            logger.error(f"  [XGBoost] Failed for {state}: {e}")
            logger.debug(traceback.format_exc())

        # ── LSTM ──────────────────────────────────────────────────────────────
        if not skip_lstm:
            try:
                result = train_lstm(
                    train_raw, val_raw, test_raw, state, ARTIFACTS_MODELS
                )
                model_results.append(result)
            except Exception as e:
                logger.error(f"  [LSTM] Failed for {state}: {e}")
                logger.debug(traceback.format_exc())
        else:
            logger.info("  [LSTM] Skipped.")

        if not model_results:
            logger.warning(f"  No models trained for {state}.")
            continue

        all_state_results[state] = model_results

    # ── Step 5: Model selection ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5 / 5: Selecting best models ...")
    final_results = select_best_models_all_states(all_state_results)

    # ── Save forecasts per state ──────────────────────────────────────────────
    for state, result in final_results.items():
        forecast_path = ARTIFACTS_FORECASTS / f"{state}.json"
        with open(forecast_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"  Forecast saved: {forecast_path}")

    # ── Save consolidated results ─────────────────────────────────────────────
    results_path = ARTIFACTS_RESULTS / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"  Consolidated results saved: {results_path}")

    elapsed = time.time() - start
    logger.info("=" * 70)
    logger.info(f"  Pipeline complete in {elapsed:.1f}s")
    logger.info("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train all forecasting models")
    parser.add_argument("--skip-lstm",   action="store_true", help="Skip LSTM training")
    parser.add_argument("--skip-sarima", action="store_true", help="Skip SARIMA training")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of states to process (for faster testing)")
    args = parser.parse_args()

    run_pipeline(skip_lstm=args.skip_lstm, skip_sarima=args.skip_sarima, limit=args.limit)
