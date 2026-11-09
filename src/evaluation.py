"""
Model Evaluation and Aerospace Metrics
======================================
Computes RMSE, MAE, R2, and the official NASA asymmetric scoring function
tailored for aircraft turbofan health management.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression metrics and the asymmetric NASA Scoring function.
    
    NASA Score definition:
      d = y_pred - y_true (error in cycles)
      s_i = exp(-d / 13) - 1 for d < 0 (early prediction, conservative)
      s_i = exp( d / 10) - 1 for d >= 0 (late prediction, critical safety hazard)
      Total Score = sum(s_i)
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    # NASA Asymmetric Score
    penalties = np.where(
        diff < 0,
        np.exp(-diff / 13.0) - 1.0,
        np.exp(diff / 10.0) - 1.0
    )
    nasa_score = float(np.sum(penalties))

    # Coefficient of determination R^2
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum(diff ** 2)
    r2 = float(1.0 - (ss_res / max(ss_tot, 1e-8)))

    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "NASA_Score": round(nasa_score, 1),
        "R2_Score": round(r2, 4)
    }


def format_metrics_table(results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Converts a dictionary of model results into a structured comparative table."""
    df = pd.DataFrame(results_dict).T
    df.index.name = "Model"
    return df
