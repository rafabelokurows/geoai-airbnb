def test_listings_for_hex(client):
    resp = client.get("/api/listings?hex_id=88abc1ffffff")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 2
    assert set(items[0].keys()) == {"id", "latitude", "longitude", "predicted_price", "predicted_occupancy"}


def test_listings_hex_with_one_listing(client):
    resp = client.get("/api/listings?hex_id=88abc2ffffff")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_listings_unknown_hex_returns_empty(client):
    resp = client.get("/api/listings?hex_id=88notexist")
    assert resp.status_code == 200
    assert resp.json() == []


def test_listings_missing_hex_id_param(client):
    resp = client.get("/api/listings")
    assert resp.status_code == 422


def test_listings_cache_header(client):
    resp = client.get("/api/listings?hex_id=88abc1ffffff")
    assert "max-age=3600" in resp.headers.get("cache-control", "")
