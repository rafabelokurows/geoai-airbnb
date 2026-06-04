"""Tests for geoai.config — verifies paths and constants are sane."""
from pathlib import Path


def test_db_path_ends_with_warehouse_duckdb():
    from geoai.config import DB_PATH
    assert DB_PATH.name == "warehouse.duckdb"
    assert DB_PATH.parent.name == "data"


def test_raw_dirs_are_children_of_data():
    from geoai.config import RAW_AIRBNB_DIR, RAW_OSM_DIR, DATA_DIR
    assert RAW_AIRBNB_DIR.parent.parent == DATA_DIR
    assert RAW_OSM_DIR.parent.parent == DATA_DIR


def test_airbnb_url_is_https():
    from geoai.config import AIRBNB_PORTO_URL
    assert AIRBNB_PORTO_URL.startswith("https://")
    assert "porto" in AIRBNB_PORTO_URL.lower()


def test_osm_city_is_porto():
    from geoai.config import OSM_CITY
    assert "Porto" in OSM_CITY


def test_osm_poi_tags_has_required_keys():
    from geoai.config import OSM_POI_TAGS
    assert "amenity" in OSM_POI_TAGS
    assert "tourism" in OSM_POI_TAGS
    assert "leisure" in OSM_POI_TAGS
    assert "railway" in OSM_POI_TAGS


def test_base_dir_is_project_root():
    from geoai.config import BASE_DIR
    # pyproject.toml lives at the project root
    assert (BASE_DIR / "pyproject.toml").exists()
