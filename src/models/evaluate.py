"""Evaluation metrics and the R2 quality gate."""
import numpy as np
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def _find_key(d, key):
    """Find `key` anywhere in a nested dict."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


def load_min_r2(config_path="configs/pipeline_config.yaml") -> float:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    val = _find_key(cfg, "min_r2")
    if val is None:
        raise KeyError(f"min_r2 not found in {config_path}")
    return float(val)


def passes_gate(metrics: dict, min_r2: float) -> bool:
    return metrics["r2"] >= min_r2