"""Cleaning and feature engineering for hourly demand forecasting.

build_features() is the single source of truth for features. Training and the
serving API must both call it, so the model sees identical features in both
places (no training-serving skew).

Forecasting setup: predict demand for hour t using only information available
at hour t-1 (lagged demand) plus calendar and weather for hour t. The weather
for hour t stands in for a weather forecast.
"""
import argparse

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET = "demand_mwh"
WEATHER_COLS = [
    "temperature_c",
    "humidity_pct",
    "wind_speed_kmh",
    "cloud_cover_pct",
    "solar_radiation_wm2",
]
ID_COLS = ["timestamp", "region", TARGET]


def _fill_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    """Linearly interpolate NaN runs of at most max_gap hours.

    Longer runs are left completely empty. (pandas' own `limit` argument would
    fill the first max_gap hours of a long run, which invents data.)
    """
    is_na = series.isna()
    run_id = (is_na != is_na.shift()).cumsum()
    run_length = is_na.groupby(run_id).transform("sum")  # length of each NaN run
    filled = series.interpolate(method="time", limit_area="inside")
    return filled.where(~(is_na & (run_length > max_gap)), np.nan)


def clean(raw: pd.DataFrame, max_gap_hours: int) -> pd.DataFrame:
    """Put the data on a complete hourly grid and fill short gaps.

    A complete grid matters: shift(24) only means "same hour yesterday" if
    every hour has a row. Gaps longer than max_gap_hours stay NaN, and the
    rows that depend on them are dropped later.
    """
    df = raw.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h", name="timestamp")
    df = df.reindex(full_index)

    cols = [TARGET, *WEATHER_COLS]
    n_missing = int(df[cols].isna().sum().sum())
    for col in cols:
        df[col] = _fill_short_gaps(df[col], max_gap_hours)
    logger.info(
        "Cleaning: %d missing cells before, %d left after filling gaps of <= %dh",
        n_missing,
        int(df[cols].isna().sum().sum()),
        max_gap_hours,
    )
    return df


def add_calendar_features(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Calendar features in LOCAL time: people follow local clocks, not UTC."""
    local = df.index.tz_localize("UTC").tz_convert(tz)
    df["hour"] = local.hour
    df["day_of_week"] = local.dayofweek
    df["month"] = local.month
    df["is_weekend"] = (local.dayofweek >= 5).astype(int)

    holidays = USFederalHolidayCalendar().holidays(
        start=local.min().date(), end=local.max().date()
    )
    local_days = local.normalize().tz_localize(None)
    df["is_holiday"] = local_days.isin(holidays).astype(int)

    # cyclical encoding so hour 23 is "close to" hour 0
    df["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * local.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * local.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * (local.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (local.month - 1) / 12)
    return df


def add_demand_history_features(df: pd.DataFrame, lags: list, windows: list) -> pd.DataFrame:
    """Lagged and rolling demand. Every rolling window is built on the series
    shifted by one hour, so the target hour never leaks into its own features."""
    for lag in lags:
        df[f"demand_lag_{lag}"] = df[TARGET].shift(lag)
    past = df[TARGET].shift(1)
    for w in windows:
        df[f"demand_roll_mean_{w}"] = past.rolling(w).mean()
        df[f"demand_roll_std_{w}"] = past.rolling(w).std()
    return df


def add_weather_features(df: pd.DataFrame, base_temp_c: float) -> pd.DataFrame:
    """Demand is V-shaped in temperature: heating below, cooling above a base."""
    df["heating_degrees"] = (base_temp_c - df["temperature_c"]).clip(lower=0)
    df["cooling_degrees"] = (df["temperature_c"] - base_temp_c).clip(lower=0)
    return df


def build_features(raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Raw demand + weather -> model-ready feature table (one row per hour)."""
    f = config["features"]
    df = clean(raw, f["max_interp_gap_hours"])
    df = add_calendar_features(df, f["timezone"])
    df = add_demand_history_features(df, f["lags"], f["rolling_windows"])
    df = add_weather_features(df, f["base_temp_c"])
    df["region"] = config["data"]["eia_region"]

    before = len(df)
    df = df.dropna().reset_index()
    logger.info("Dropped %d rows with incomplete history/NaNs; %d rows remain", before - len(df), len(df))

    feature_cols = [c for c in df.columns if c not in ID_COLS]
    return df[[*ID_COLS, *feature_cols]]


def get_feature_columns(df: pd.DataFrame) -> list:
    """Names of the model input columns (everything except id and target)."""
    return [c for c in df.columns if c not in ID_COLS]


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    """Chronological split. Never shuffle time series: the model must be
    evaluated on data that comes after everything it trained on."""
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the feature table")
    parser.add_argument("--raw", help="raw CSV (default: data.raw_path)")
    parser.add_argument("--out", help="output parquet (default: data.processed_path)")
    args = parser.parse_args()

    config = load_config()
    raw_path = PROJECT_ROOT / (args.raw or config["data"]["raw_path"])
    out_path = PROJECT_ROOT / (args.out or config["data"]["processed_path"])

    raw = pd.read_csv(raw_path, parse_dates=["timestamp"])
    features = build_features(raw, config)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    logger.info("Saved %d rows x %d columns to %s", *features.shape, out_path)

    s = config["split"]
    for name, part in zip(["train", "val", "test"], time_split(features, s["train_frac"], s["val_frac"])):
        logger.info("%-5s %6d rows  %s -> %s", name, len(part), part["timestamp"].min(), part["timestamp"].max())


if __name__ == "__main__":
    main()
