import pytest


def test_hexagons_default_mode_price(client):
    resp = client.get("/api/hexagons")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 2
    assert set(items[0].keys()) == {"hex_id", "value", "listing_count"}


def test_hexagons_mode_occupancy(client):
    resp = client.get("/api/hexagons?mode=occupancy")
    assert resp.status_code == 200
    # hex '88abc1ffffff' has avg_occupancy=0.775
    values = {item["hex_id"]: item["value"] for item in resp.json()}
    assert abs(values["88abc1ffffff"] - 0.775) < 0.001


def test_hexagons_mode_revenue(client):
    resp = client.get("/api/hexagons?mode=revenue")
    assert resp.status_code == 200
    values = {item["hex_id"]: item["value"] for item in resp.json()}
    assert abs(values["88abc2ffffff"] - 1440.0) < 0.01


def test_hexagons_invalid_mode(client):
    resp = client.get("/api/hexagons?mode=invalid")
    assert resp.status_code == 422


def test_hexagons_cache_header(client):
    resp = client.get("/api/hexagons")
    assert "max-age=3600" in resp.headers.get("cache-control", "")


def test_hex_detail_valid(client):
    resp = client.get("/api/hexagons/88abc1ffffff")
    assert resp.status_code == 200
    body = resp.json()
    assert body["hex_id"] == "88abc1ffffff"
    assert abs(body["avg_price"] - 110.0) < 0.01
    assert body["listing_count"] == 2
    assert set(body.keys()) == {
        "hex_id", "avg_price", "avg_occupancy", "avg_revenue",
        "listing_count", "avg_walkability_score", "avg_restaurant_density",
        "avg_dist_city_center_km", "avg_competition_score",
    }


def test_hex_detail_not_found(client):
    resp = client.get("/api/hexagons/88notexist")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "hex not found"


def test_hex_detail_cache_header(client):
    resp = client.get("/api/hexagons/88abc1ffffff")
    assert "max-age=3600" in resp.headers.get("cache-control", "")
