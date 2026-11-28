"""Copy features from the offline store into the online store."""
import pandas as pd
from feast import FeatureStore

from feature_store.retrieve import DATA_PATH, REPO_PATH

if __name__ == "__main__":
    ts = pd.to_datetime(pd.read_parquet(DATA_PATH, columns=["timestamp"])["timestamp"], utc=True)
    start, end = ts.min().to_pydatetime(), (ts.max() + pd.Timedelta(hours=1)).to_pydatetime()
    FeatureStore(repo_path=str(REPO_PATH)).materialize(start_date=start, end_date=end)
    print(f"Materialized {start} -> {end}")
