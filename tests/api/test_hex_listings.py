import polars as pl
from fastapi.testclient import TestClient

from geoai.api.deps import app_state
from geoai.api.main import create_app


def _seed_state():
    app_state.listings_df = pl.DataFrame({
        "id": [1, 2, 3],
        "h3_cell_r8": ["abc123", "abc123", "xyz999"],
        "price": [80.0, 120.0, 60.0],
        "predicted_occupancy": [0.7, 0.8, 0.6],
        "latitude": [38.7, 38.7, 38.7],
        "longitude": [-9.1, -9.1, -9.1],
        "room_type": ["Entire home/apt"] * 3,
        "neighbourhood": ["A", "A", "B"],
        "predicted_price": [None, None, None],
        "estimated_annual_revenue": [None, None, None],
    })


def test_hex_listings_returns_filtered_results():
    _seed_state()
    client = TestClient(create_app(load_state=False))
    resp = client.get("/api/hex/abc123/listings")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert all(d["price"] is not None for d in data)


def test_hex_listings_returns_empty_for_unknown_cell():
    _seed_state()
    client = TestClient(create_app(load_state=False))
    resp = client.get("/api/hex/nonexistent/listings")
    assert resp.status_code == 200
    assert resp.json() == []
