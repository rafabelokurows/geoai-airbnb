import polars as pl
import pytest
from fastapi.testclient import TestClient

from geoai.api.deps import app_state
from geoai.api.main import create_app


@pytest.fixture(autouse=True)
def mock_state():
    # listing 1: price 50, predicted 100 → 50% underpriced → qualifies
    # listing 2: price 90, predicted 100 → 10% underpriced → does NOT qualify
    # listing 3: price 110, predicted 100 → overpriced → does NOT qualify
    app_state.listings_df = pl.DataFrame({
        "id": [1, 2, 3],
        "latitude": [41.1, 41.2, 41.3],
        "longitude": [-8.6, -8.61, -8.62],
        "price": [50.0, 90.0, 110.0],
        "room_type": ["Entire home/apt"] * 3,
        "h3_cell_r8": ["abc"] * 3,
        "predicted_price": [100.0, 100.0, 100.0],
        "predicted_occupancy": [0.6, 0.7, 0.5],
        "estimated_annual_revenue": [21900.0, 25550.0, 18250.0],
    })
    yield
    app_state.listings_df = None


client = TestClient(create_app(load_state=False))


def test_opportunities_only_returns_underpriced():
    r = client.get("/api/opportunities")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["listing_id"] == "1"


def test_opportunities_schema():
    r = client.get("/api/opportunities")
    item = r.json()[0]
    assert set(item.keys()) == {
        "listing_id", "latitude", "longitude",
        "actual_price", "predicted_price",
        "opportunity_gap", "estimated_uplift_annual"
    }


def test_opportunities_gap_value():
    r = client.get("/api/opportunities")
    assert r.json()[0]["opportunity_gap"] == pytest.approx(50.0)
