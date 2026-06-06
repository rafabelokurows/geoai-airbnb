import polars as pl
import pytest
from fastapi.testclient import TestClient

from geoai.api.deps import app_state
from geoai.api.main import create_app


def _make_listings_df(n: int = 5) -> pl.DataFrame:
    return pl.DataFrame({
        "id": list(range(1, n + 1)),
        "latitude": [41.1 + i * 0.01 for i in range(n)],
        "longitude": [-8.6 + i * 0.01 for i in range(n)],
        "price": [80.0 + i * 10 for i in range(n)],
        "room_type": ["Entire home/apt"] * n,
        "h3_cell_r8": ["abc"] * n,
        "predicted_price": [85.0 + i * 10 for i in range(n)],
        "predicted_occupancy": [0.6 + i * 0.05 for i in range(n)],
        "estimated_annual_revenue": [18000.0 + i * 1000 for i in range(n)],
    })


@pytest.fixture(autouse=True)
def mock_state():
    app_state.listings_df = _make_listings_df(5)
    yield
    app_state.listings_df = None


client = TestClient(create_app(load_state=False))


def test_listings_returns_list():
    r = client.get("/api/listings")
    assert r.status_code == 200
    data = r.json()
    assert "listings" in data
    assert "total" in data
    assert data["total"] == 5


def test_listings_schema():
    r = client.get("/api/listings")
    listing = r.json()["listings"][0]
    assert set(listing.keys()) == {
        "id", "latitude", "longitude", "price", "room_type",
        "predicted_price", "predicted_occupancy", "estimated_annual_revenue"
    }


def test_listings_pagination():
    r = client.get("/api/listings?limit=2&offset=1")
    data = r.json()
    assert len(data["listings"]) == 2
    assert data["listings"][0]["id"] == 2
