def test_shap_global_price(client):
    resp = client.get("/api/shap/global?model=price")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 2
    assert set(items[0].keys()) == {"feature", "importance"}
    # must be sorted descending by importance
    importances = [i["importance"] for i in items]
    assert importances == sorted(importances, reverse=True)


def test_shap_global_occupancy(client):
    resp = client.get("/api/shap/global?model=occupancy")
    assert resp.status_code == 200
    features = {i["feature"] for i in resp.json()}
    assert "walkability_score" in features


def test_shap_global_invalid_model(client):
    resp = client.get("/api/shap/global?model=banana")
    assert resp.status_code == 422


def test_shap_global_default_is_price(client):
    resp = client.get("/api/shap/global")
    assert resp.status_code == 200


def test_shap_global_cache_header(client):
    resp = client.get("/api/shap/global?model=price")
    assert "max-age=3600" in resp.headers.get("cache-control", "")


def test_shap_hex_valid(client):
    resp = client.get("/api/shap/88abc1ffffff?model=price")
    assert resp.status_code == 200
    body = resp.json()
    assert body["hex_id"] == "88abc1ffffff"
    assert isinstance(body["base_value"], float)
    assert isinstance(body["drivers"], list)
    assert set(body["drivers"][0].keys()) == {"feature", "avg_impact"}
    # sorted by abs(avg_impact) descending
    impacts = [abs(d["avg_impact"]) for d in body["drivers"]]
    assert impacts == sorted(impacts, reverse=True)


def test_shap_hex_not_found(client):
    resp = client.get("/api/shap/88notexist?model=price")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "hex not found"


def test_shap_hex_cache_header(client):
    resp = client.get("/api/shap/88abc1ffffff?model=price")
    assert "max-age=3600" in resp.headers.get("cache-control", "")
