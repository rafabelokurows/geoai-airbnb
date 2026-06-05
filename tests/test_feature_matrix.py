import numpy as np
import polars as pl
import pytest

from geoai.models.features import (
    NUMERIC_FEATURE_COLS,
    prepare_X_y_price,
    prepare_X_y_occupancy,
)


def _make_df(n: int = 20) -> pl.DataFrame:
    """Synthetic DataFrame matching build_feature_matrix() output schema."""
    rng = np.random.default_rng(42)
    data = {col: rng.uniform(0, 10, n).tolist() for col in NUMERIC_FEATURE_COLS}
    data["id"] = list(range(n))
    data["room_type"] = ["Entire home/apt"] * n
    data["target_price"] = rng.uniform(50, 300, n).tolist()
    data["target_occupancy"] = rng.uniform(0.1, 0.9, n).tolist()
    return pl.DataFrame(data)


def test_prepare_X_y_price_shape():
    df = _make_df(20)
    X, y_log, feature_names = prepare_X_y_price(df)
    assert X.shape[0] == 20
    assert y_log.shape[0] == 20
    assert len(feature_names) == X.shape[1]


def test_prepare_X_y_price_log_transform():
    df = _make_df(10)
    X, y_log, _ = prepare_X_y_price(df)
    prices = df["target_price"].to_numpy()
    np.testing.assert_allclose(y_log, np.log(prices), rtol=1e-5)


def test_prepare_X_y_price_no_nulls():
    df = _make_df(10)
    X, y_log, _ = prepare_X_y_price(df)
    assert not np.isnan(X).any()
    assert not np.isnan(y_log).any()


def test_prepare_X_y_occupancy_includes_price():
    df = _make_df(10)
    _, _, price_features = prepare_X_y_price(df)
    _, _, occ_features = prepare_X_y_occupancy(df)
    assert "price" in occ_features
    assert len(occ_features) > len(price_features)


def test_prepare_X_y_occupancy_target_in_unit_interval():
    df = _make_df(10)
    _, y, _ = prepare_X_y_occupancy(df)
    assert (y >= 0).all() and (y <= 1).all()
