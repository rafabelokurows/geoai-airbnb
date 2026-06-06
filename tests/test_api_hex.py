import polars as pl
import pytest
from fastapi.testclient import TestClient

from geoai.api.deps import app_state
from geoai.api.main import create_app


@pytest.fixture(autouse=True)
def mock_state():
    app_state.hex_df = pl.DataFrame({
        "h3_cell": ["881eaad2edfffff", "881eaad2ebfffff"],
        "listing_count": [10, 5],
        "avg_price": [90.0, 110.0],
        "avg_occupancy": [0.65, 0.72],
        "avg_walkability": [70.0, 80.0],
        "avg_revenue": [21352.5, 28908.0],
    })
    yield
    app_state.hex_df = None


client = TestClient(create_app(load_state=False))


def test_hex_aggregates_returns_list():
    r = client.get("/api/hex-aggregates")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 2


def test_hex_aggregates_schema():
    r = client.get("/api/hex-aggregates")
    cell = r.json()[0]
    assert set(cell.keys()) == {"h3_cell", "listing_count", "avg_price", "avg_occupancy", "avg_revenue"}


def test_hex_aggregates_values():
    r = client.get("/api/hex-aggregates")
    cell = next(c for c in r.json() if c["h3_cell"] == "881eaad2edfffff")
    assert cell["listing_count"] == 10
    assert cell["avg_price"] == pytest.approx(90.0)
