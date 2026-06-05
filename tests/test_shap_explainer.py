import numpy as np
import polars as pl
import lightgbm as lgb
import pytest

from geoai.models.features import NUMERIC_FEATURE_COLS, prepare_X_y_price
from geoai.explainability.shap_explainer import (
    build_explainer,
    compute_global_importance,
    explain_listing,
)


def _train_small_model():
    rng = np.random.default_rng(42)
    n = 200
    data = {col: rng.uniform(0, 10, n).tolist() for col in NUMERIC_FEATURE_COLS}
    data["id"] = list(range(n))
    data["room_type"] = ["Entire home/apt"] * n
    acc = rng.integers(1, 8, n)
    data["accommodates"] = acc.tolist()
    data["target_price"] = (50.0 + acc * 30.0 + rng.normal(0, 5, n)).tolist()
    data["target_occupancy"] = rng.uniform(0.1, 0.9, n).tolist()
    df = pl.DataFrame(data)
    X, y_log, feature_names = prepare_X_y_price(df)
    model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
    model.fit(X, y_log)
    return model, X, list(feature_names)


def test_build_explainer_returns_tree_explainer():
    import shap
    model, X, feature_names = _train_small_model()
    explainer = build_explainer(model, X)
    assert isinstance(explainer, shap.TreeExplainer)


def test_global_importance_shape_and_columns():
    model, X, feature_names = _train_small_model()
    explainer = build_explainer(model, X)
    importance_df = compute_global_importance(explainer, X, feature_names)
    assert set(importance_df.columns) == {"feature", "importance"}
    assert len(importance_df) == len(feature_names)


def test_global_importance_sorted_descending():
    model, X, feature_names = _train_small_model()
    explainer = build_explainer(model, X)
    importance_df = compute_global_importance(explainer, X, feature_names)
    vals = importance_df["importance"].to_list()
    assert vals == sorted(vals, reverse=True)


def test_global_importance_nonnegative():
    model, X, feature_names = _train_small_model()
    explainer = build_explainer(model, X)
    importance_df = compute_global_importance(explainer, X, feature_names)
    assert (importance_df["importance"] >= 0).all()


def test_explain_listing_returns_expected_keys():
    model, X, feature_names = _train_small_model()
    explainer = build_explainer(model, X)
    result = explain_listing(explainer, X[0], feature_names)
    assert "predicted_value" in result
    assert "base_value" in result
    assert "drivers" in result
    assert isinstance(result["drivers"], list)
    assert len(result["drivers"]) > 0
    driver = result["drivers"][0]
    assert "feature" in driver
    assert "impact" in driver


def test_explain_listing_top_n():
    model, X, feature_names = _train_small_model()
    explainer = build_explainer(model, X)
    result = explain_listing(explainer, X[0], feature_names, top_n=5)
    assert len(result["drivers"]) == 5


def test_explain_listing_drivers_sorted_by_abs_impact():
    model, X, feature_names = _train_small_model()
    explainer = build_explainer(model, X)
    result = explain_listing(explainer, X[0], feature_names)
    impacts = [abs(d["impact"]) for d in result["drivers"]]
    assert impacts == sorted(impacts, reverse=True)
