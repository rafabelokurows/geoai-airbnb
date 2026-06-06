import numpy as np
import lightgbm as lgb
import polars as pl
import pytest
from fastapi.testclient import TestClient

from geoai.api.deps import app_state
from geoai.api.main import create_app
from geoai.models.features import NUMERIC_FEATURE_COLS, prepare_X_y_price
from geoai.explainability.shap_explainer import build_explainer


def _build_test_explainer():
    rng = np.random.default_rng(42)
    n = 100
    data = {col: rng.uniform(0, 10, n).tolist() for col in NUMERIC_FEATURE_COLS}
    data["id"] = list(range(n))
    data["room_type"] = ["Entire home/apt"] * n
    acc = rng.integers(1, 8, n)
    data["accommodates"] = acc.tolist()
    data["target_price"] = (50.0 + acc * 30.0 + rng.normal(0, 5, n)).tolist()
    data["target_occupancy"] = rng.uniform(0.1, 0.9, n).tolist()
    df = pl.DataFrame(data)
    X, _, feature_names = prepare_X_y_price(df)
    model = lgb.LGBMRegressor(n_estimators=20, random_state=42, verbose=-1)
    model.fit(X, np.log(df["target_price"].to_numpy()))
    explainer = build_explainer(model, X)
    return explainer, X, list(feature_names)


@pytest.fixture(autouse=True)
def mock_state():
    explainer, X, features = _build_test_explainer()
    app_state.price_explainer = explainer
    app_state.X_price = X
    app_state.price_features = features
    app_state.listing_id_to_idx = {i: i for i in range(len(X))}
    yield
    app_state.price_explainer = None
    app_state.X_price = None
    app_state.price_features = []
    app_state.listing_id_to_idx = {}


client = TestClient(create_app(load_state=False))


def test_explain_returns_200():
    r = client.get("/api/listings/0/explain")
    assert r.status_code == 200


def test_explain_schema():
    r = client.get("/api/listings/0/explain")
    data = r.json()
    assert set(data.keys()) == {"listing_id", "predicted_price", "base_value", "drivers"}
    assert data["listing_id"] == 0
    assert len(data["drivers"]) == 5
    driver = data["drivers"][0]
    assert "feature" in driver and "impact" in driver


def test_explain_404_for_unknown_listing():
    r = client.get("/api/listings/99999/explain")
    assert r.status_code == 404


def test_explain_predicted_price_positive():
    r = client.get("/api/listings/0/explain")
    assert r.json()["predicted_price"] > 0
