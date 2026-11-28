"""Offline (training) and online (serving) feature retrieval."""
from pathlib import Path

import numpy as np
import pandas as pd
from feast import FeatureStore

ROOT = Path(__file__).resolve().parents[1]
REPO_PATH = ROOT / "feature_store" / "feature_repo"
DATA_PATH = ROOT / "data/processed/energy_features.parquet"
FEATURE_VIEW = "energy_features"
SERVICE = "energy_demand_v1"


def _store(store):
    return store or FeatureStore(repo_path=str(REPO_PATH))


def get_training_data(store: FeatureStore | None = None) -> pd.DataFrame:
    """Full hourly history for training, plus the demand target.

    Feature names and the offline source come from the Feast registry.
    Feast's point-in-time join on a single hourly series is quadratic
    (about 8.5 MB per entity row), so full history is read straight from
    the registered source instead; see get_point_in_time_sample for the
    Feast join check.
    """
    store = _store(store)
    fv = store.get_feature_view(FEATURE_VIEW)
    feats = [f.name for f in fv.features]
    path = getattr(fv.batch_source, "path", None) or DATA_PATH
    df = pd.read_parquet(path, columns=["region", "timestamp"] + feats + ["demand_mwh"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def get_point_in_time_sample(n: int = 40, store: FeatureStore | None = None) -> pd.DataFrame:
    """Feast point-in-time join for n rows spread across the history (keep n small)."""
    store = _store(store)
    labels = pd.read_parquet(DATA_PATH, columns=["region", "timestamp", "demand_mwh"])
    idx = np.linspace(0, len(labels) - 1, n).astype(int)
    entity_df = labels.iloc[idx].rename(columns={"timestamp": "event_timestamp"}).copy()
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"], utc=True)
    return store.get_historical_features(
        entity_df=entity_df, features=store.get_feature_service(SERVICE)
    ).to_df()


def get_online_features(region_value: str, store: FeatureStore | None = None) -> dict:
    store = _store(store)
    return store.get_online_features(
        features=store.get_feature_service(SERVICE),
        entity_rows=[{"region": region_value}],
    ).to_dict()


if __name__ == "__main__":
    df = get_training_data()
    print("Offline (full history):", df.shape)
    print(df.head(2).T)
    sample = get_point_in_time_sample()
    print("PIT sample:", sample.shape)
    print("Online:", get_online_features(df["region"].iloc[0]))