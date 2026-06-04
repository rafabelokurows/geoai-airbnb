import pytest
import duckdb
from pathlib import Path

from geoai.database.warehouse import init_warehouse, get_connection


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
    assert {"listings", "poi_features"}.issubset(tables)
    con2.close()


def test_get_connection_returns_live_connection(tmp_path):
    db_path = tmp_path / "test.duckdb"
    init_warehouse(db_path)
    con = get_connection(db_path)
    result = con.execute("SELECT 42").fetchone()[0]
    assert result == 42
    con.close()


def test_calendar_table_exists(tmp_path):
    con = init_warehouse(tmp_path / "test.duckdb")
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert "calendar" in tables
    con.close()


def test_listing_features_table_exists(tmp_path):
    con = init_warehouse(tmp_path / "test.duckdb")
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert "listing_features" in tables
    con.close()


def test_hex_aggregates_table_exists(tmp_path):
    con = init_warehouse(tmp_path / "test.duckdb")
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert "hex_aggregates" in tables
    con.close()
