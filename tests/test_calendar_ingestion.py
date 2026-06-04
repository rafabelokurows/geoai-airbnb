import polars as pl
import pytest

from geoai.ingestion.calendar import clean_calendar, load_calendar_into_db
from geoai.database.warehouse import init_warehouse


def _sample_calendar() -> pl.DataFrame:
    return pl.DataFrame({
        "listing_id": [1, 2, 3, 4],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "available": ["t", "f", "t", None],
        "price": ["$85.00", "$120.50", "$200.00", "$50.00"],
        "minimum_nights": [2, 1, 3, 2],
        "maximum_nights": [30, 30, 30, 30],
    })


def test_clean_calendar_parses_available_flag():
    df = clean_calendar(_sample_calendar())
    assert df["available"][0] is True
    assert df["available"][1] is False


def test_clean_calendar_parses_price():
    df = clean_calendar(_sample_calendar())
    assert df["price"][0] == pytest.approx(85.0)
    assert df["price"][1] == pytest.approx(120.5)


def test_clean_calendar_parses_date():
    df = clean_calendar(_sample_calendar())
    import datetime
    assert df["date"][0] == datetime.date(2024, 1, 1)


def test_clean_calendar_drops_null_date_or_id():
    df = pl.DataFrame({
        "listing_id": [1, None, 3],
        "date": ["2024-01-01", "2024-01-02", None],
        "available": ["t", "t", "f"],
        "price": ["$100.00", "$50.00", "$75.00"],
        "minimum_nights": [1, 1, 1],
        "maximum_nights": [30, 30, 30],
    })
    cleaned = clean_calendar(df)
    assert len(cleaned) == 1


def test_load_calendar_into_db_returns_count(tmp_path, monkeypatch):
    sample = _sample_calendar()
    monkeypatch.setattr("geoai.ingestion.calendar._download_raw", lambda url, dest: None)
    monkeypatch.setattr("polars.read_csv", lambda path, **kw: sample)
    count = load_calendar_into_db(db_path=tmp_path / "test.duckdb")
    assert count == 4  # available=None row kept — drop_nulls only on listing_id and date
