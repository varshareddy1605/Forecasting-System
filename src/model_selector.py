"""
model_selector.py
-----------------
Compares evaluation metrics from all trained models for a given state
and selects the best one based on lowest RMSE on the test set.

Logs and returns a structured results dictionary.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def select_best_model(model_results: List[Dict], state: str) -> Dict:
    """
    Selects and returns the model with the lowest RMSE.
    """
    valid = [r for r in model_results if r.get("metrics", {}).get("rmse") is not None]

    if not valid:
        logger.warning(f"[Selector] No valid models for {state}. Returning first.")
        return model_results[0] if model_results else {}

    best = min(valid, key=lambda r: r["metrics"]["rmse"])

    logger.info(
        f"[Selector] {state} — Best model: {best['model_name']} "
        f"(RMSE={best['metrics']['rmse']:.4f})"
    )

    logger.info(f"[Selector] {state} — All model metrics:")
    for r in sorted(valid, key=lambda x: x["metrics"]["rmse"]):
        logger.info(
            f"  {r['model_name']:10s} | RMSE={r['metrics']['rmse']:.4f} "
            f"| MAE={r['metrics']['mae']:.4f} "
            f"| MAPE={r['metrics']['mape']:.4f}"
        )

    result = {
        "state": state,
        "best_model": best["model_name"],
        "forecast": best["forecast"],
        "metrics": best["metrics"],
        "all_metrics": {
            r["model_name"]: r["metrics"] for r in valid
        },
    }
    return result


def select_best_models_all_states(
    all_state_results: Dict[str, List[Dict]]
) -> Dict[str, Dict]:
    """
    Runs model selection for all states and returns a mapped dictionary.
    """
    final = {}
    for state, model_results in all_state_results.items():
        final[state] = select_best_model(model_results, state)
    return final
