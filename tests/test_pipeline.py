"""Tests for the data checks and the cleaning step, on made-up rows."""
import pandas as pd

from coldstore.clean import clean
from coldstore.data_checks import check_duplicate_rows, check_sentinel_value, check_stuck_sensor

NO_ISSUES = {
    "leaking_column": {"present": False},
    "unit_mixup": {"present": False},
    "sentinel_value": {"present": False},
    "duplicate_rows": {"present": False},
    "stuck_sensor": {"present": False},
    "target_typos": {"present": False},
    "spelling_variants": {"present": False},
}


def table(n=30):
    dates = pd.date_range("2026-01-01", periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({
        "record_id": [f"R{i:03d}" for i in range(n)],
        "date": dates,
        "humidity_pct": [80 + i % 10 for i in range(n)],
        "door_openings_per_day": [i % 7 for i in range(n)],
        "chamber": ["Chamber 1"] * n,
        "leak": range(n),
        "spoilage_pct": [5.0 + i % 4 for i in range(n)],
    })


def test_sentinel_is_found_and_cleaned():
    df = table()
    df.loc[:5, "humidity_pct"] = -999
    found = check_sentinel_value(df, ["humidity_pct"])
    assert found == {"present": True, "column": "humidity_pct", "value": -999, "count": 6}
    cleaned = clean(df, {**NO_ISSUES, "sentinel_value": found})
    assert cleaned["humidity_pct"].isna().sum() == 6


def test_stuck_sensor_dates():
    df = table()
    df.loc[10:24, "door_openings_per_day"] = 29
    found = check_stuck_sensor(df, ["door_openings_per_day"], min_run=10)
    assert found["start_date"] == "2026-01-11"
    assert found["end_date"] == "2026-01-25"
    cleaned = clean(df, {**NO_ISSUES, "stuck_sensor": found})
    assert cleaned["door_openings_per_day"].isna().sum() == 15


def test_leak_is_dropped_and_duplicates_counted():
    df = pd.concat([table(), table().iloc[:3]])
    assert check_duplicate_rows(df)["count"] == 3
    cleaned = clean(df, {**NO_ISSUES, "leaking_column": {"present": True, "column": "leak"}})
    assert "leak" not in cleaned.columns
