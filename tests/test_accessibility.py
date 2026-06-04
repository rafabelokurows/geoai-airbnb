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


def test_compute_accessibility_returns_all_columns():
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
    expected = {
        "dist_city_center_km", "dist_nearest_metro_km", "dist_nearest_station_km",
        "dist_nearest_supermarket_km",
        "dist_livraria_lello_km", "dist_torre_clerigos_km", "dist_ribeira_km",
        "dist_ponte_luis_km", "dist_mercado_bolhao_km", "dist_jardins_cristal_km",
        "dist_airport_km", "travel_time_airport_min",
    }
    assert expected.issubset(set(result.columns))


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


def test_compute_accessibility_landmark_distances_reasonable():
    # Listing at Porto center — all landmarks should be within 3km
    listings = pl.DataFrame({
        "id": [1], "latitude": [41.14961], "longitude": [-8.61099]
    })
    pois = pl.DataFrame(schema={"osm_id": pl.Utf8, "poi_type": pl.Utf8, "poi_subtype": pl.Utf8, "latitude": pl.Float64, "longitude": pl.Float64})
    result = compute_accessibility(listings, pois)
    for col in ["dist_livraria_lello_km", "dist_torre_clerigos_km", "dist_ribeira_km",
                "dist_ponte_luis_km", "dist_mercado_bolhao_km", "dist_jardins_cristal_km"]:
        assert result[col][0] < 3.0, f"{col} unexpectedly large"


def test_compute_accessibility_airport_distance():
    listings = pl.DataFrame({
        "id": [1], "latitude": [41.14961], "longitude": [-8.61099]
    })
    pois = pl.DataFrame(schema={"osm_id": pl.Utf8, "poi_type": pl.Utf8, "poi_subtype": pl.Utf8, "latitude": pl.Float64, "longitude": pl.Float64})
    result = compute_accessibility(listings, pois)
    # Porto center to airport ~11km straight-line
    assert result["dist_airport_km"][0] == pytest.approx(11.0, abs=3.0)
    # travel time = dist * 1.3 / 40 * 60 — must be positive and reasonable
    assert result["travel_time_airport_min"][0] == pytest.approx(
        result["dist_airport_km"][0] * 1.3 / 40.0 * 60.0, abs=0.01
    )


def test_compute_accessibility_supermarket_null_when_no_pois():
    listings = pl.DataFrame({
        "id": [1], "latitude": [41.14961], "longitude": [-8.61099]
    })
    pois = pl.DataFrame(schema={"osm_id": pl.Utf8, "poi_type": pl.Utf8, "poi_subtype": pl.Utf8, "latitude": pl.Float64, "longitude": pl.Float64})
    result = compute_accessibility(listings, pois)
    assert result["dist_nearest_supermarket_km"][0] is None
