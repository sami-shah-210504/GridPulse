"""Data source and feature schema, read from the parquet so names never drift."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from feast import Field, FileSource
from feast.types import Bool, Float64, Int64

DATA_PATH = Path(__file__).resolve().parents[2] / "data/processed/energy_features.parquet"
NON_FEATURES = {"timestamp", "region", "demand_mwh"}  # ids + target


def _dtype(t):
    if pa.types.is_boolean(t):
        return Bool
    if pa.types.is_integer(t):
        return Int64
    return Float64


source = FileSource(
    name="energy_features_source",
    path=str(DATA_PATH),
    timestamp_field="timestamp",
)

FEATURE_FIELDS = [
    Field(name=f.name, dtype=_dtype(f.type))
    for f in pq.read_schema(DATA_PATH)
    if f.name not in NON_FEATURES
]
