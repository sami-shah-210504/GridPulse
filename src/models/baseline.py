"""Seasonal naive baseline: predict demand = same hour last week."""
import pandas as pd

from src.models.evaluate import compute_metrics


def score_baseline(df: pd.DataFrame, lag_col: str = "demand_lag_168") -> dict:
    return compute_metrics(df["demand_mwh"], df[lag_col])