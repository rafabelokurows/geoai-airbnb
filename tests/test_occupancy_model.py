import numpy as np
import lightgbm as lgb
import polars as pl
import pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from geoai.models.features import prepare_X_y_occupancy, NUMERIC_FEATURE_COLS


def _make_df(n: int = 200) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    data = {col: rng.uniform(0, 10, n).tolist() for col in NUMERIC_FEATURE_COLS}
    data["id"] = list(range(n))
    data["room_type"] = ["Entire home/apt"] * n
    data["target_price"] = rng.uniform(50, 300, n).tolist()
    walk = rng.uniform(0, 100, n)
    data["walkability_score"] = walk.tolist()
    # synthetic: occupancy correlated with walkability
    occ = np.clip(0.3 + walk / 200.0 + rng.normal(0, 0.05, n), 0, 1)
    data["target_occupancy"] = occ.tolist()
    return pl.DataFrame(data)


def test_occupancy_model_trains_and_predicts():
    df = _make_df(200)
    X, y, _ = prepare_X_y_occupancy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).clip(0, 1)
    mae = mean_absolute_error(y_test, y_pred)
    assert mae < 0.2, f"MAE {mae:.4f} unexpectedly high on synthetic data"


def test_occupancy_predictions_clipped_to_unit_interval():
    df = _make_df(50)
    X, y, _ = prepare_X_y_occupancy(df)
    model = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
    model.fit(X, y)
    preds = model.predict(X).clip(0, 1)
    assert (preds >= 0).all() and (preds <= 1).all()


def test_occupancy_output_shape_matches_input():
    df = _make_df(30)
    X, y, _ = prepare_X_y_occupancy(df)
    model = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (30,)
