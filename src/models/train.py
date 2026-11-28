"""Compare the baseline, XGBoost, LightGBM and Random Forest on the validation set.

Training data comes from the Feast feature store (feature_store.retrieve).
The test set is split off here but not touched; it is used once, at the end.
"""
import logging
import time

import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from feature_store.retrieve import get_training_data
from src.models.baseline import score_baseline
from src.models.evaluate import compute_metrics, load_min_r2, passes_gate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ID_COLS = ["region", "timestamp"]
TARGET = "demand_mwh"
SEED = 42


def chronological_split(df, train=0.70, val=0.15):
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    i, j = int(n * train), int(n * (train + val))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


def get_xy(df):
    return df.drop(columns=ID_COLS + [TARGET]), df[TARGET]


def build_models():
    return {
        "xgboost": XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, random_state=SEED, n_jobs=-1, early_stopping_rounds=30,
        ),
        "lightgbm": lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1,
        ),
    }


def fit(name, model, X_train, y_train, X_val, y_val):
    if name == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "lightgbm":
        model.fit(
            X_train, y_train, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
    else:
        model.fit(X_train, y_train)
    return model


def main():
    df = get_training_data()
    train_df, val_df, test_df = chronological_split(df)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        log.info("%-5s %6d rows  %s -> %s", name, len(part),
                 part["timestamp"].min(), part["timestamp"].max())

    X_train, y_train = get_xy(train_df)
    X_val, y_val = get_xy(val_df)
    log.info("Features: %d", X_train.shape[1])

    results = {
        "baseline_lag168": score_baseline(val_df, "demand_lag_168"),
        "baseline_lag24": score_baseline(val_df, "demand_lag_24"),
        "baseline_lag1": score_baseline(val_df, "demand_lag_1"),
    }
    best_name, best_model = None, None

    for name, model in build_models().items():
        start = time.time()
        fit(name, model, X_train, y_train, X_val, y_val)
        secs = time.time() - start
        metrics = compute_metrics(y_val, model.predict(X_val))
        metrics["train_secs"] = secs
        results[name] = metrics
        log.info("%-14s r2=%.4f rmse=%.1f (%.1fs)", name, metrics["r2"], metrics["rmse"], secs)
        if best_name is None or metrics["r2"] > results[best_name]["r2"]:
            best_name, best_model = name, model

    print("\nValidation results")
    print(pd.DataFrame(results).T.round(4).to_string())

    min_r2 = load_min_r2()
    best = results[best_name]
    log.info("Best model: %s (val R2 %.4f)", best_name, best["r2"])
    log.info("Gate (R2 >= %.2f): %s", min_r2, "PASS" if passes_gate(best, min_r2) else "FAIL")
    best_base = max(v["r2"] for k, v in results.items() if k.startswith("baseline"))
    log.info("Beats best baseline (R2 %.4f): %s", best_base, best["r2"] > best_base)


if __name__ == "__main__":
    main()