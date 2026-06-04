import pytest
import polars as pl
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from pathlib import Path

from geoai.ingestion.osm import process_pois, load_osm_into_db
from geoai.database.warehouse import init_warehouse


def _sample_gdf() -> gpd.GeoDataFrame:
    data = {
        "amenity": ["restaurant", "bar", None, None],
        "tourism": [None, None, "museum", None],
        "leisure": [None, None, None, "park"],
        "railway": [None, None, None, None],
        "name": ["Café Majestic", "Hard Club", "Museu do Vinho", "Jardim do Palácio"],
        "geometry": [
            Point(-8.611, 41.149),
            Point(-8.615, 41.150),
            Point(-8.612, 41.148),
            Polygon([(-8.610, 41.147), (-8.609, 41.147), (-8.609, 41.148), (-8.610, 41.147)]),
        ],
    }
    index = pd.MultiIndex.from_tuples(
        [("node", 111), ("node", 222), ("node", 333), ("way", 444)],
        names=["element_type", "osmid"],
    )
    return gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:4326", index=index)


def test_process_pois_returns_polars_dataframe():
    gdf = _sample_gdf()
    result = process_pois(gdf)
    assert isinstance(result, pl.DataFrame)


def test_process_pois_correct_row_count():
    gdf = _sample_gdf()
    result = process_pois(gdf)
    assert len(result) == 4


def test_process_pois_extracts_poi_type():
    gdf = _sample_gdf()
    result = process_pois(gdf)
    types = set(result["poi_type"].to_list())
    assert "amenity" in types
    assert "tourism" in types
    assert "leisure" in types


def test_process_pois_polygon_uses_centroid():
    gdf = _sample_gdf()
    result = process_pois(gdf)
    # way/444 is a Polygon — centroid should be inside polygon bounds
    way_row = result.filter(pl.col("osm_id") == "way_444")
    assert len(way_row) == 1
    lat = way_row["latitude"][0]
    lon = way_row["longitude"][0]
    assert 41.147 <= lat <= 41.148
    assert -8.610 <= lon <= -8.609


def test_process_pois_generates_unique_osm_ids():
    gdf = _sample_gdf()
    result = process_pois(gdf)
    ids = result["osm_id"].to_list()
    assert len(ids) == len(set(ids))


def test_load_osm_into_db_returns_count(tmp_path, monkeypatch):
    sample_gdf = _sample_gdf()

    monkeypatch.setattr(
        "geoai.ingestion.osm._fetch_osm",
        lambda city, tags: sample_gdf,
    )

    db_path = tmp_path / "test.duckdb"
    count = load_osm_into_db(db_path=db_path)
    assert count == 4
