import datetime
import polars as pl
import pytest

from geoai.features.occupancy import compute_occupancy


def _calendar(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame({
        "listing_id": [r["listing_id"] for r in rows],
        "date": [r["date"] for r in rows],
        "available": [r["available"] for r in rows],
    })


def test_compute_occupancy_returns_correct_columns():
    cal = _calendar([
        {"listing_id": 1, "date": datetime.date(2024, 12, 1), "available": False},
        {"listing_id": 1, "date": datetime.date(2024, 12, 2), "available": True},
    ])
    result = compute_occupancy(cal)
    assert "listing_id" in result.columns
    assert "occupancy_rate_30d" in result.columns
    assert "occupancy_rate_90d" in result.columns
    assert "occupancy_rate_365d" in result.columns


def test_compute_occupancy_fully_booked():
    dates = [datetime.date(2024, 12, 1) + datetime.timedelta(days=i) for i in range(30)]
    cal = _calendar([{"listing_id": 1, "date": d, "available": False} for d in dates])
    result = compute_occupancy(cal)
    assert result.filter(pl.col("listing_id") == 1)["occupancy_rate_30d"][0] == pytest.approx(1.0)


def test_compute_occupancy_fully_available():
    dates = [datetime.date(2024, 12, 1) + datetime.timedelta(days=i) for i in range(30)]
    cal = _calendar([{"listing_id": 1, "date": d, "available": True} for d in dates])
    result = compute_occupancy(cal)
    assert result.filter(pl.col("listing_id") == 1)["occupancy_rate_30d"][0] == pytest.approx(0.0)


def test_compute_occupancy_mixed():
    dates = [datetime.date(2024, 12, 1) + datetime.timedelta(days=i) for i in range(30)]
    avail = [i % 2 == 0 for i in range(30)]
    cal = _calendar([{"listing_id": 1, "date": d, "available": a} for d, a in zip(dates, avail)])
    result = compute_occupancy(cal)
    assert result.filter(pl.col("listing_id") == 1)["occupancy_rate_30d"][0] == pytest.approx(0.5, abs=0.05)


def test_compute_occupancy_no_data_for_listing_returns_null():
    cal = _calendar([
        {"listing_id": 1, "date": datetime.date(2024, 12, 1), "available": False},
    ])
    result = compute_occupancy(cal)
    assert len(result.filter(pl.col("listing_id") == 2)) == 0
