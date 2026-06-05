import numpy as np
import lightgbm as lgb
import polars as pl
import pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from geoai.models.features import prepare_X_y_price, NUMERIC_FEATURE_COLS


def _make_df(n: int = 200) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    data = {col: rng.uniform(0, 10, n).tolist() for col in NUMERIC_FEATURE_COLS}
    data["id"] = list(range(n))
    data["room_type"] = ["Entire home/apt"] * n
    accommodates = rng.integers(1, 8, n)
    data["accommodates"] = accommodates.tolist()
    # synthetic: price = 50 + accommodates*30 + noise
    data["target_price"] = (50.0 + accommodates * 30.0 + rng.normal(0, 5, n)).tolist()
    data["target_occupancy"] = rng.uniform(0.1, 0.9, n).tolist()
    return pl.DataFrame(data)


def test_price_model_trains_and_predicts():
    df = _make_df(200)
    X, y_log, _ = prepare_X_y_price(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    y_pred = np.exp(model.predict(X_test))
    y_true = np.exp(y_test)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    assert rmse < 50, f"RMSE {rmse:.2f} unexpectedly high on synthetic data"


def test_price_model_output_shape_matches_input():
    df = _make_df(50)
    X, y_log, _ = prepare_X_y_price(df)
    model = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
    model.fit(X, y_log)
    preds = model.predict(X)
    assert preds.shape == (50,)


def test_price_predictions_are_positive():
    df = _make_df(50)
    X, y_log, _ = prepare_X_y_price(df)
    model = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
    model.fit(X, y_log)
    preds_price = np.exp(model.predict(X))
    assert (preds_price > 0).all()
