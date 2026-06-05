import duckdb
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_db(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))

    con.execute("""
        CREATE TABLE listing_predictions (
            listing_id BIGINT, h3_cell_r8 VARCHAR,
            predicted_price DOUBLE, predicted_occupancy DOUBLE,
            predicted_revenue DOUBLE, latitude DOUBLE, longitude DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO listing_predictions VALUES
        (1, '88abc1ffffff', 100.0, 0.75, 2250.0, 41.147, -8.611),
        (2, '88abc1ffffff', 120.0, 0.80, 2880.0, 41.148, -8.612),
        (3, '88abc2ffffff', 80.0,  0.60, 1440.0, 41.150, -8.620)
    """)

    con.execute("""
        CREATE TABLE hex_aggregates (
            h3_cell_r8 VARCHAR PRIMARY KEY,
            listing_count BIGINT,
            avg_price DOUBLE, avg_occupancy DOUBLE, avg_revenue DOUBLE,
            avg_walkability_score DOUBLE, avg_restaurant_density DOUBLE,
            avg_dist_city_center_km DOUBLE, avg_competition_score DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO hex_aggregates VALUES
        ('88abc1ffffff', 2, 110.0, 0.775, 2565.0, 85.0, 0.8, 0.5, 6.0),
        ('88abc2ffffff', 1,  80.0, 0.600, 1440.0, 70.0, 0.5, 1.2, 3.0)
    """)

    con.execute("""
        CREATE TABLE shap_global (
            model VARCHAR, feature VARCHAR, importance DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO shap_global VALUES
        ('price',     'walkability_score',   0.42),
        ('price',     'restaurant_density',  0.38),
        ('occupancy', 'walkability_score',   0.30),
        ('occupancy', 'restaurant_density',  0.25)
    """)

    con.execute("""
        CREATE TABLE hex_shap (
            h3_cell_r8 VARCHAR, model VARCHAR, feature VARCHAR,
            avg_impact DOUBLE, base_value DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO hex_shap VALUES
        ('88abc1ffffff', 'price',     'walkability_score',  15.0, 87.0),
        ('88abc1ffffff', 'price',     'restaurant_density', 10.0, 87.0),
        ('88abc1ffffff', 'occupancy', 'walkability_score',  0.05, 0.71),
        ('88abc1ffffff', 'occupancy', 'restaurant_density', 0.03, 0.71)
    """)

    con.close()
    return db_path


@pytest.fixture
def client(test_db):
    from geoai.api.deps import get_db
    from geoai.api.main import app

    test_con = duckdb.connect(str(test_db), read_only=True)
    app.dependency_overrides[get_db] = lambda: test_con

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
    test_con.close()
