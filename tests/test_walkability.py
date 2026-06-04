import polars as pl
import pytest

from geoai.features.walkability import compute_walkability


def _feature_row(**overrides) -> pl.DataFrame:
    defaults = {
        "listing_id": 1,
        "restaurants_500m": 5,
        "bars_500m": 2,
        "cafes_500m": 3,
        "dist_nearest_metro_km": 0.2,
        "dist_city_center_km": 1.0,
        "parks_500m": 1,
        "supermarkets_1km": 1,
    }
    defaults.update(overrides)
    return pl.DataFrame({k: [v] for k, v in defaults.items()})


def test_compute_walkability_returns_score_column():
    result = compute_walkability(_feature_row())
    assert "walkability_score" in result.columns
    assert "listing_id" in result.columns


def test_compute_walkability_score_in_range():
    result = compute_walkability(_feature_row())
    score = result["walkability_score"][0]
    assert 0.0 <= score <= 100.0


def test_compute_walkability_perfect_score():
    result = compute_walkability(_feature_row(
        restaurants_500m=10, bars_500m=5, cafes_500m=5,
        dist_nearest_metro_km=0.0, dist_city_center_km=0.0,
        parks_500m=3, supermarkets_1km=2
    ))
    assert result["walkability_score"][0] == pytest.approx(100.0, abs=0.1)


def test_compute_walkability_zero_score():
    result = compute_walkability(_feature_row(
        restaurants_500m=0, bars_500m=0, cafes_500m=0,
        dist_nearest_metro_km=2.0, dist_city_center_km=10.0,
        parks_500m=0, supermarkets_1km=0
    ))
    assert result["walkability_score"][0] == pytest.approx(0.0, abs=0.1)


def test_compute_walkability_null_metro_distance():
    result = compute_walkability(_feature_row(dist_nearest_metro_km=None))
    score = result["walkability_score"][0]
    assert 0.0 <= score <= 100.0
