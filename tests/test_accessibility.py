import numpy as np
import polars as pl
import pytest

from geoai.features.accessibility import haversine_km, compute_accessibility


def test_haversine_km_same_point():
    d = haversine_km(
        np.array([41.14961]), np.array([-8.61099]),
        41.14961, -8.61099
    )
    assert d[0] == pytest.approx(0.0, abs=1e-6)


def test_haversine_km_known_distance():
    # Porto to Lisbon: ~274 km
    d = haversine_km(
        np.array([41.14961]), np.array([-8.61099]),
        38.71667, -9.13333
    )
    assert d[0] == pytest.approx(274.0, abs=5.0)


def test_compute_accessibility_returns_three_columns():
    listings = pl.DataFrame({
        "id": [1], "latitude": [41.14961], "longitude": [-8.61099]
    })
    metro_pois = pl.DataFrame({
        "osm_id": ["node_1"], "poi_type": ["railway"],
        "poi_subtype": ["subway_entrance"],
        "latitude": [41.150], "longitude": [-8.612],
    })
    station_pois = pl.DataFrame({
        "osm_id": ["node_2"], "poi_type": ["railway"],
        "poi_subtype": ["station"],
        "latitude": [41.160], "longitude": [-8.620],
    })
    pois = pl.concat([metro_pois, station_pois])
    result = compute_accessibility(listings, pois)
    assert "dist_city_center_km" in result.columns
    assert "dist_nearest_metro_km" in result.columns
    assert "dist_nearest_station_km" in result.columns


def test_compute_accessibility_city_center_distance():
    listings = pl.DataFrame({
        "id": [1], "latitude": [41.14961], "longitude": [-8.61099]
    })
    pois = pl.DataFrame({
        "osm_id": ["n1"], "poi_type": ["railway"], "poi_subtype": ["subway_entrance"],
        "latitude": [41.15], "longitude": [-8.61],
    })
    result = compute_accessibility(listings, pois)
    assert result["dist_city_center_km"][0] == pytest.approx(0.0, abs=0.01)


def test_compute_accessibility_no_metro_returns_null():
    listings = pl.DataFrame({
        "id": [1], "latitude": [41.14961], "longitude": [-8.61099]
    })
    pois = pl.DataFrame(schema={"osm_id": pl.Utf8, "poi_type": pl.Utf8, "poi_subtype": pl.Utf8, "latitude": pl.Float64, "longitude": pl.Float64})
    result = compute_accessibility(listings, pois)
    assert result["dist_nearest_metro_km"][0] is None
