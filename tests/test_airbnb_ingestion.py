import pytest
import polars as pl
from pathlib import Path

from geoai.ingestion.airbnb import clean_listings, load_airbnb_into_db
from geoai.database.warehouse import init_warehouse


def _sample_raw_df() -> pl.DataFrame:
    return pl.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Cozy Studio", "Modern Flat", "River View", "Bad Listing"],
        "latitude": [41.1496, 41.1500, 41.1480, None],
        "longitude": [-8.6110, -8.6120, -8.6100, None],
        "neighbourhood_cleansed": ["Cedofeita", "Bonfim", "Ribeira", "Unknown"],
        "room_type": ["Entire home/apt", "Private room", "Entire home/apt", "Private room"],
        "property_type": ["Apartment", "Apartment", "House", "Apartment"],
        "accommodates": [2, 1, 4, 1],
        "bedrooms": [1.0, 1.0, 2.0, 1.0],
        "beds": [1.0, 1.0, 2.0, 1.0],
        "price": ["$85.00", "$1,200.00", "$120.00", "$50.00"],
        "minimum_nights": [2, 1, 3, 1],
        "maximum_nights": [30, 365, 30, 30],
        "availability_30": [10, 20, 5, 0],
        "availability_60": [20, 40, 10, 0],
        "availability_90": [30, 60, 15, 0],
        "availability_365": [120, 240, 60, 0],
        "number_of_reviews": [15, 30, 8, 0],
        "review_scores_rating": [4.8, 4.5, 4.9, None],
        "reviews_per_month": [1.2, 2.5, 0.8, None],
        "host_id": [1001, 1002, 1003, 9999],
        "host_name": ["Ana", "João", "Maria", "Ghost"],
        "host_is_superhost": ["t", "f", "t", "f"],
        "amenities": ['["Wifi","Kitchen"]', '["Wifi"]', '["Wifi","Pool"]', "[]"],
        "last_scraped": ["2024-12-22", "2024-12-22", "2024-12-22", "2024-12-22"],
    })


def test_clean_listings_parses_price_removes_symbols():
    df = _sample_raw_df()
    cleaned = clean_listings(df)
    assert cleaned["price"].dtype == pl.Float64
    assert cleaned["price"][0] == pytest.approx(85.0)
    assert cleaned["price"][1] == pytest.approx(1200.0)


def test_clean_listings_parses_superhost_flag():
    df = _sample_raw_df()
    cleaned = clean_listings(df)
    assert cleaned["host_is_superhost"][0] is True
    assert cleaned["host_is_superhost"][1] is False


def test_clean_listings_renames_neighbourhood_column():
    df = _sample_raw_df()
    cleaned = clean_listings(df)
    assert "neighbourhood" in cleaned.columns
    assert "neighbourhood_cleansed" not in cleaned.columns


def test_clean_listings_drops_rows_with_null_coordinates():
    df = _sample_raw_df()
    cleaned = clean_listings(df)
    assert len(cleaned) == 3  # row 4 has None lat/lon — dropped
    assert cleaned["id"].to_list() == [1, 2, 3]


def test_load_airbnb_into_db_returns_correct_count(tmp_path, monkeypatch):
    sample = _sample_raw_df()
    fake_path = tmp_path / "listings.csv"

    monkeypatch.setattr(
        "geoai.ingestion.airbnb._download_raw",
        lambda url, dest_dir: fake_path,
    )
    monkeypatch.setattr(
        "geoai.ingestion.airbnb._read_raw",
        lambda path: sample,
    )

    db_path = tmp_path / "test.duckdb"
    init_warehouse(db_path)
    count = load_airbnb_into_db(db_path=db_path)
    assert count == 3  # 3 valid rows (row 4 dropped during cleaning)


def test_load_airbnb_into_db_stores_correct_price(tmp_path, monkeypatch):
    sample = _sample_raw_df()
    fake_path = tmp_path / "listings.csv"

    monkeypatch.setattr(
        "geoai.ingestion.airbnb._download_raw",
        lambda url, dest_dir: fake_path,
    )
    monkeypatch.setattr(
        "geoai.ingestion.airbnb._read_raw",
        lambda path: sample,
    )

    db_path = tmp_path / "test.duckdb"
    load_airbnb_into_db(db_path=db_path)
    import duckdb
    con = duckdb.connect(str(db_path))
    price = con.execute("SELECT price FROM listings WHERE id = 2").fetchone()[0]
    con.close()
    assert price == pytest.approx(1200.0)


def test_clean_listings_superhost_none_for_unknown_value():
    df = pl.DataFrame({
        "id": [1],
        "latitude": [41.15],
        "longitude": [-8.61],
        "neighbourhood_cleansed": ["X"],
        "room_type": ["Entire home/apt"],
        "property_type": ["Apartment"],
        "accommodates": [2],
        "bedrooms": [1.0],
        "beds": [1.0],
        "price": ["$80.00"],
        "minimum_nights": [1],
        "maximum_nights": [30],
        "availability_30": [5],
        "availability_60": [10],
        "availability_90": [15],
        "availability_365": [60],
        "number_of_reviews": [5],
        "review_scores_rating": [4.5],
        "reviews_per_month": [0.5],
        "host_id": [1001],
        "host_name": ["Ana"],
        "host_is_superhost": ["unknown"],
        "amenities": ["[]"],
        "last_scraped": ["2024-12-22"],
        "name": ["Test"],
    })
    cleaned = clean_listings(df)
    assert cleaned["host_is_superhost"][0] is None
