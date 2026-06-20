import polars as pl
from fastapi.testclient import TestClient

from geoai.api.deps import app_state
from geoai.api.main import create_app


def _seed_state():
    app_state.listings_df = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "neighbourhood": ["Alfama", "Alfama", "Bairro Alto", "Bairro Alto"],
        "estimated_annual_revenue": [10000.0, 20000.0, 5000.0, 7000.0],
        "latitude": [38.7, 38.7, 38.7, 38.7],
        "longitude": [-9.1, -9.1, -9.1, -9.1],
        "price": [100.0, 120.0, 80.0, 90.0],
        "room_type": ["Entire home/apt"] * 4,
        "h3_cell_r8": ["a", "a", "b", "b"],
        "neighbourhood": ["Alfama", "Alfama", "Bairro Alto", "Bairro Alto"],
        "predicted_price": [None, None, None, None],
        "predicted_occupancy": [None, None, None, None],
    })


def test_neighbourhoods_returns_ranked_list():
    _seed_state()
    client = TestClient(create_app(load_state=False))
    resp = client.get("/api/neighbourhoods")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["neighbourhood"] == "Alfama"
    assert data[0]["avg_revenue"] == 15000.0
    assert data[0]["listing_count"] == 2
    assert data[1]["neighbourhood"] == "Bairro Alto"


def test_neighbourhoods_sorted_descending():
    _seed_state()
    client = TestClient(create_app(load_state=False))
    resp = client.get("/api/neighbourhoods")
    data = resp.json()
    revenues = [d["avg_revenue"] for d in data]
    assert revenues == sorted(revenues, reverse=True)
