def test_stats_returns_correct_shape(client):
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"avg_price", "avg_occupancy", "median_revenue", "listing_count"}


def test_stats_listing_count(client):
    resp = client.get("/api/stats")
    assert resp.json()["listing_count"] == 3


def test_stats_avg_price(client):
    resp = client.get("/api/stats")
    avg = resp.json()["avg_price"]
    # test DB: (100 + 120 + 80) / 3 = 100.0
    assert abs(avg - 100.0) < 0.01


def test_stats_cache_header(client):
    resp = client.get("/api/stats")
    assert "max-age=3600" in resp.headers.get("cache-control", "")
