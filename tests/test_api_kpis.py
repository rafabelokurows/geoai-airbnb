import polars as pl
import pytest
from fastapi.testclient import TestClient

from geoai.api.deps import app_state
from geoai.api.main import create_app


@pytest.fixture(autouse=True)
def mock_state():
    app_state.listings_df = pl.DataFrame({
        "id": [1, 2, 3],
        "latitude": [41.1, 41.2, 41.3],
        "longitude": [-8.6, -8.61, -8.62],
        "price": [80.0, 100.0, 120.0],
        "room_type": ["Entire home/apt"] * 3,
        "h3_cell_r8": ["abc"] * 3,
        "predicted_price": [85.0, 95.0, None],
        "predicted_occupancy": [0.6, 0.7, None],
        "estimated_annual_revenue": [18615.0, 24272.5, None],
    })
    yield
    app_state.listings_df = None


client = TestClient(create_app(load_state=False))


def test_kpis_keys():
    r = client.get("/api/kpis")
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {
        "listing_count", "listings_with_predictions",
        "avg_price", "avg_occupancy", "median_annual_revenue"
    }


def test_kpis_listing_count():
    r = client.get("/api/kpis")
    assert r.json()["listing_count"] == 3


def test_kpis_predictions_count():
    r = client.get("/api/kpis")
    assert r.json()["listings_with_predictions"] == 2
