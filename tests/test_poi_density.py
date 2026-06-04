import math

import polars as pl
import pytest

from geoai.features.poi_density import compute_poi_density, count_within_radius


def _make_listings(coords: list[tuple[float, float]]) -> pl.DataFrame:
    return pl.DataFrame({
        "id": list(range(len(coords))),
        "latitude": [c[0] for c in coords],
        "longitude": [c[1] for c in coords],
    })


def _make_pois(entries: list[dict]) -> pl.DataFrame:
    return pl.DataFrame({
        "osm_id": [f"n{i}" for i in range(len(entries))],
        "poi_type": [e["poi_type"] for e in entries],
        "poi_subtype": [e["poi_subtype"] for e in entries],
        "latitude": [e["lat"] for e in entries],
        "longitude": [e["lon"] for e in entries],
    })


def test_count_within_radius_returns_correct_count():
    listing_lat, listing_lon = 41.14961, -8.61099
    pois = _make_pois([
        {"poi_type": "amenity", "poi_subtype": "restaurant", "lat": 41.150, "lon": -8.611},   # ~0.11km
        {"poi_type": "amenity", "poi_subtype": "restaurant", "lat": 41.151, "lon": -8.612},   # ~0.17km
        {"poi_type": "amenity", "poi_subtype": "restaurant", "lat": 41.200, "lon": -8.650},   # ~7km
    ])
    restaurants = pois.filter(pl.col("poi_subtype").is_in(["restaurant", "fast_food"]))
    count = count_within_radius(listing_lat, listing_lon, restaurants, radius_km=0.5)
    assert count == 2


def test_count_within_radius_empty_pois():
    count = count_within_radius(41.14961, -8.61099, _make_pois([]), radius_km=0.5)
    assert count == 0


def test_compute_poi_density_returns_correct_columns():
    listings = _make_listings([(41.14961, -8.61099)])
    pois = _make_pois([
        {"poi_type": "amenity", "poi_subtype": "restaurant", "lat": 41.150, "lon": -8.611},
        {"poi_type": "leisure", "poi_subtype": "park", "lat": 41.149, "lon": -8.610},
    ])
    result = compute_poi_density(listings, pois)
    expected_cols = {
        "listing_id", "restaurants_250m", "restaurants_500m", "bars_500m",
        "cafes_500m", "supermarkets_1km", "attractions_1km", "museums_2km", "parks_500m",
        "amenity_density_1km", "restaurant_density",
    }
    assert expected_cols.issubset(set(result.columns))


def test_compute_poi_density_amenity_density_and_restaurant_density():
    listings = _make_listings([(41.14961, -8.61099)])
    pois = _make_pois([
        {"poi_type": "amenity", "poi_subtype": "restaurant", "lat": 41.150,  "lon": -8.611},  # ~0.11km
        {"poi_type": "amenity", "poi_subtype": "restaurant", "lat": 41.151,  "lon": -8.612},  # ~0.17km
        {"poi_type": "amenity", "poi_subtype": "bar",        "lat": 41.149,  "lon": -8.610},  # ~0.05km
        {"poi_type": "leisure", "poi_subtype": "park",       "lat": 41.148,  "lon": -8.609},  # ~0.20km
        {"poi_type": "amenity", "poi_subtype": "restaurant", "lat": 41.200,  "lon": -8.650},  # ~7km — far
    ])
    result = compute_poi_density(listings, pois)
    # 2 restaurants + 1 bar + 1 park within 1km
    assert result["amenity_density_1km"][0] == 4
    # 2 restaurants within 500m → density = 2 / (π * 0.25)
    assert result["restaurant_density"][0] == pytest.approx(2 / math.pi / 0.25, abs=0.01)


def test_compute_poi_density_counts_correctly():
    listings = _make_listings([(41.14961, -8.61099)])
    pois = _make_pois([
        {"poi_type": "amenity", "poi_subtype": "restaurant", "lat": 41.1498, "lon": -8.6109},  # ~0.03km
        {"poi_type": "amenity", "poi_subtype": "restaurant", "lat": 41.152,  "lon": -8.613},   # ~0.28km
        {"poi_type": "amenity", "poi_subtype": "restaurant", "lat": 41.158,  "lon": -8.618},   # ~0.9km
    ])
    result = compute_poi_density(listings, pois)
    assert result["restaurants_250m"][0] == 1
    assert result["restaurants_500m"][0] == 2
