import polars as pl
import pytest

from geoai.features.competition import compute_competition


def _listings(data: list[dict]) -> pl.DataFrame:
    return pl.DataFrame({
        "id": [d["id"] for d in data],
        "latitude": [d["lat"] for d in data],
        "longitude": [d["lon"] for d in data],
        "price": [d["price"] for d in data],
        "neighbourhood": [d["neighbourhood"] for d in data],
    })


def test_compute_competition_excludes_self():
    listings = _listings([
        {"id": 1, "lat": 41.14961, "lon": -8.61099, "price": 100.0, "neighbourhood": "A"},
        {"id": 2, "lat": 41.1500,  "lon": -8.6110,  "price": 120.0, "neighbourhood": "A"},
    ])
    result = compute_competition(listings)
    row1 = result.filter(pl.col("listing_id") == 1)
    assert row1["listings_500m"][0] == 1
    assert row1["listings_1km"][0] == 1


def test_compute_competition_counts_within_radius():
    listings = _listings([
        {"id": 1, "lat": 41.14961, "lon": -8.61099, "price": 100.0, "neighbourhood": "A"},
        {"id": 2, "lat": 41.1500,  "lon": -8.6110,  "price": 120.0, "neighbourhood": "A"},  # ~0.1km
        {"id": 3, "lat": 41.200,   "lon": -8.650,   "price": 200.0, "neighbourhood": "B"},  # ~7km
    ])
    result = compute_competition(listings)
    row1 = result.filter(pl.col("listing_id") == 1)
    assert row1["listings_500m"][0] == 1
    assert row1["listings_1km"][0] == 1


def test_compute_competition_avg_price_500m():
    listings = _listings([
        {"id": 1, "lat": 41.14961, "lon": -8.61099, "price": 100.0, "neighbourhood": "A"},
        {"id": 2, "lat": 41.1500,  "lon": -8.6110,  "price": 200.0, "neighbourhood": "A"},
        {"id": 3, "lat": 41.1505,  "lon": -8.6115,  "price": 300.0, "neighbourhood": "A"},
    ])
    result = compute_competition(listings)
    row1 = result.filter(pl.col("listing_id") == 1)
    assert row1["avg_price_500m"][0] == pytest.approx(250.0, abs=1.0)


def test_compute_competition_median_price_neighbourhood():
    listings = _listings([
        {"id": 1, "lat": 41.14961, "lon": -8.61099, "price": 100.0, "neighbourhood": "A"},
        {"id": 2, "lat": 41.150,   "lon": -8.611,   "price": 200.0, "neighbourhood": "A"},
        {"id": 3, "lat": 41.151,   "lon": -8.612,   "price": 300.0, "neighbourhood": "A"},
        {"id": 4, "lat": 41.200,   "lon": -8.650,   "price": 500.0, "neighbourhood": "B"},
    ])
    result = compute_competition(listings)
    row1 = result.filter(pl.col("listing_id") == 1)
    assert row1["median_price_neighbourhood"][0] == pytest.approx(200.0, abs=1.0)


def test_compute_competition_no_neighbours_avg_price_is_null():
    listings = _listings([
        {"id": 1, "lat": 41.14961, "lon": -8.61099, "price": 100.0, "neighbourhood": "A"},
        {"id": 2, "lat": 41.500,   "lon": -9.000,   "price": 200.0, "neighbourhood": "B"},
    ])
    result = compute_competition(listings)
    row1 = result.filter(pl.col("listing_id") == 1)
    assert row1["avg_price_500m"][0] is None
