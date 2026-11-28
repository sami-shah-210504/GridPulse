"""Run after `feast apply` and `python -m feature_store.materialize`."""
import numpy as np
import pandas as pd
import pytest

from feature_store.retrieve import (
    DATA_PATH,
    get_online_features,
    get_point_in_time_sample,
    get_training_data,
)

NON_FEATURES = ("timestamp", "region", "demand_mwh")


@pytest.fixture(scope="module")
def raw():
    return pd.read_parquet(DATA_PATH).sort_values("timestamp").reset_index(drop=True)


def feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURES]


def test_entity_key_and_timestamp(raw):
    assert {"region", "timestamp"} <= set(raw.columns)
    assert raw["region"].notna().all()
    assert not raw.duplicated(["region", "timestamp"]).any()


def test_no_nulls(raw):
    assert raw.isna().sum().sum() == 0


def test_hourly_freshness(raw):
    gaps = raw["timestamp"].diff().dropna()
    assert gaps.max() <= pd.Timedelta(hours=6)


def test_offline_matches_parquet(raw):
    train = get_training_data()
    cols = feature_cols(raw)
    assert len(train) == len(raw)
    np.testing.assert_allclose(train[cols].to_numpy(float), raw[cols].to_numpy(float))


def test_online_matches_latest_row(raw):
    out = get_online_features(raw["region"].iloc[0])
    last = raw.iloc[-1]
    for c in feature_cols(raw):
        assert out[c][0] is not None
        assert out[c][0] == pytest.approx(float(last[c]))


def test_point_in_time_sample_matches_parquet(raw):
    sample = get_point_in_time_sample(n=40)
    ts = pd.to_datetime(sample["event_timestamp"], utc=True)
    ref = raw.assign(timestamp=pd.to_datetime(raw["timestamp"], utc=True)).set_index("timestamp")
    cols = feature_cols(raw)
    np.testing.assert_allclose(sample[cols].to_numpy(float), ref.loc[ts, cols].to_numpy(float))
