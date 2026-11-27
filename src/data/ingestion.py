"""Data ingestion: hourly electricity demand (EIA) + hourly weather (Open-Meteo).

Both sources are requested in UTC so their timestamps line up. The output is
one tidy CSV with a UTC timestamp, the target (demand_mwh) and weather columns.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import PROJECT_ROOT, get_eia_key, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
EIA_PAGE_SIZE = 5000  # EIA returns at most 5000 rows per request

# Open-Meteo variable name -> our column name
WEATHER_VARS = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity_pct",
    "wind_speed_10m": "wind_speed_kmh",
    "cloud_cover": "cloud_cover_pct",
    "shortwave_radiation": "solar_radiation_wm2",
}


def _get_with_retries(url: str, params: dict, retries: int = 3, timeout: int = 60):
    """GET with simple exponential backoff. Raises after the last failure."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            logger.warning("Request failed (attempt %d/%d): %s", attempt, retries, exc)
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)


def fetch_eia_demand(region: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """Fetch hourly demand (type D) for one balancing authority, paginating."""
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": region,
        "facets[type][]": "D",  # D = demand
        "start": f"{start}T00",
        "end": f"{end}T23",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": EIA_PAGE_SIZE,
    }
    rows, offset = [], 0
    while True:
        params["offset"] = offset
        payload = _get_with_retries(EIA_URL, params).json()["response"]
        page = payload.get("data", [])
        rows.extend(page)
        logger.info("EIA: fetched %d rows (offset %d)", len(page), offset)
        if len(page) < EIA_PAGE_SIZE:
            break
        offset += EIA_PAGE_SIZE

    if not rows:
        raise ValueError(f"EIA returned no data for region={region} {start}..{end}")

    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            # EIA 'period' looks like 2023-01-01T08 and is UTC
            "timestamp": pd.to_datetime(df["period"], format="%Y-%m-%dT%H"),
            "demand_mwh": pd.to_numeric(df["value"], errors="coerce"),
        }
    )
    return out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch hourly historical weather in UTC from Open-Meteo (no API key)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "UTC",
    }
    hourly = _get_with_retries(WEATHER_URL, params).json()["hourly"]
    df = pd.DataFrame(hourly).rename(columns=WEATHER_VARS)
    df["timestamp"] = pd.to_datetime(df.pop("time"))
    logger.info("Open-Meteo: fetched %d rows", len(df))
    return df[["timestamp", *WEATHER_VARS.values()]]


def build_raw_dataset(config: dict, api_key: str) -> pd.DataFrame:
    d = config["data"]
    demand = fetch_eia_demand(d["eia_region"], d["start_date"], d["end_date"], api_key)
    weather = fetch_weather(d["latitude"], d["longitude"], d["start_date"], d["end_date"])
    # left join: keep every demand hour, even if weather is missing (validation will flag it)
    merged = demand.merge(weather, on="timestamp", how="left")
    logger.info("Merged dataset: %d rows, %d columns", *merged.shape)
    return merged


def save_raw(df: pd.DataFrame, path: str) -> Path:
    out = PROJECT_ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Saved raw data to %s", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw demand + weather data")
    parser.add_argument("--start", help="override start_date (YYYY-MM-DD)")
    parser.add_argument("--end", help="override end_date (YYYY-MM-DD)")
    args = parser.parse_args()

    config = load_config()
    if args.start:
        config["data"]["start_date"] = args.start
    if args.end:
        config["data"]["end_date"] = args.end

    df = build_raw_dataset(config, get_eia_key())
    save_raw(df, config["data"]["raw_path"])


if __name__ == "__main__":
    main()
