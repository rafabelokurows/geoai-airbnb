import pytest
import duckdb
from pathlib import Path
import tempfile

from geoai.database.warehouse import init_warehouse


def test_init_warehouse_creates_both_tables(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = init_warehouse(db_path)
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    assert "listings" in tables
    assert "poi_features" in tables
    con.close()


def test_listings_has_required_columns(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = init_warehouse(db_path)
    cols = {row[0] for row in con.execute("DESCRIBE listings").fetchall()}
    required = {"id", "latitude", "longitude", "price", "room_type", "neighbourhood"}
    assert required.issubset(cols)
    con.close()


def test_poi_features_has_required_columns(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = init_warehouse(db_path)
    cols = {row[0] for row in con.execute("DESCRIBE poi_features").fetchall()}
    required = {"osm_id", "poi_type", "poi_subtype", "latitude", "longitude"}
    assert required.issubset(cols)
    con.close()


def test_init_warehouse_is_idempotent(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = init_warehouse(db_path)
    con.close()
    # Second call must not raise
    con2 = init_warehouse(db_path)
    tables = {row[0] for row in con2.execute("SHOW TABLES").fetchall()}
    assert len(tables) == 2
    con2.close()
