"""Data validation: schema checks (Pandera) + data-quality checks.

Hard failures raise DataValidationError so the pipeline stops and a bad dataset
never reaches training. A JSON report is always written for auditing.
"""
import argparse
import json

import pandas as pd
import pandera as pa

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_MISSING_RATIO = 0.05   # >5% missing in any column -> fail
MAX_GAP_HOURS = 6          # a hole longer than 6 hours in the series -> fail
REPORT_PATH = PROJECT_ROOT / "reports" / "validation_report.json"

# Types and plausible physical ranges. nullable=True here because missing
# values are judged by the ratio check below, not rejected one by one.
SCHEMA = pa.DataFrameSchema(
    {
        "timestamp": pa.Column(pa.DateTime, unique=True, nullable=False, coerce=True),
        "demand_mwh": pa.Column(float, pa.Check.in_range(1, 100_000), nullable=True, coerce=True),
        "temperature_c": pa.Column(float, pa.Check.in_range(-40, 60), nullable=True, coerce=True),
        "humidity_pct": pa.Column(float, pa.Check.in_range(0, 100), nullable=True, coerce=True),
        "wind_speed_kmh": pa.Column(float, pa.Check.in_range(0, 250), nullable=True, coerce=True),
        "cloud_cover_pct": pa.Column(float, pa.Check.in_range(0, 100), nullable=True, coerce=True),
        "solar_radiation_wm2": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
    },
    strict=True,  # reject unexpected columns
)


class DataValidationError(Exception):
    """Raised when the dataset fails a mandatory check."""


def validate(df: pd.DataFrame) -> dict:
    """Run all checks. Returns a report dict; raises DataValidationError on failure."""
    failures = []
    report = {
        "rows": int(len(df)),
        "columns": list(df.columns),
    }

    # 1. Schema: column names, types, unique timestamps, value ranges
    try:
        df = SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        failures.append(f"schema: {len(exc.failure_cases)} violation(s)")
        report["schema_failures"] = exc.failure_cases.head(20).astype(str).to_dict("records")
        _finish(report, failures)  # cannot run later checks on a broken schema

    # 2. Missing-value ratio per column
    missing = df.isna().mean()
    report["missing_ratio"] = {c: round(float(v), 4) for c, v in missing.items()}
    for col, ratio in missing.items():
        if ratio > MAX_MISSING_RATIO:
            failures.append(f"missing: {col} is {ratio:.1%} null (limit {MAX_MISSING_RATIO:.0%})")

    # 3. Time continuity: hourly data with no long holes
    ts = df["timestamp"].sort_values()
    gaps = ts.diff().dropna()
    max_gap = gaps.max() if len(gaps) else pd.Timedelta(0)
    report["max_gap_hours"] = float(max_gap / pd.Timedelta(hours=1))
    if max_gap > pd.Timedelta(hours=MAX_GAP_HOURS):
        failures.append(f"gap: largest hole is {report['max_gap_hours']:.0f}h (limit {MAX_GAP_HOURS}h)")

    report["time_range"] = [str(ts.min()), str(ts.max())] if len(ts) else None
    _finish(report, failures)
    return report


def _finish(report: dict, failures: list) -> None:
    report["passed"] = not failures
    report["failures"] = failures
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    if failures:
        for f in failures:
            logger.error("VALIDATION FAILED: %s", f)
        raise DataValidationError("; ".join(failures))
    logger.info("Validation passed: %d rows", report["rows"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the raw dataset")
    parser.add_argument("--path", help="CSV to validate (default: raw_path in config)")
    args = parser.parse_args()
    path = PROJECT_ROOT / (args.path or load_config()["data"]["raw_path"])
    validate(pd.read_csv(path, parse_dates=["timestamp"]))


if __name__ == "__main__":
    main()
