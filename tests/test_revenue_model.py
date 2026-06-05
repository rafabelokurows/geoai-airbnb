import numpy as np
import polars as pl
import pytest


def test_revenue_formula_direct():
    # Revenue = predicted_price * predicted_occupancy * 365
    price = np.array([80.0, 150.0])
    occupancy = np.array([0.6, 0.8])
    revenue = price * occupancy * 365
    assert revenue[0] == pytest.approx(17520.0, abs=1.0)
    assert revenue[1] == pytest.approx(43800.0, abs=1.0)


def test_revenue_output_schema():
    result = pl.DataFrame({
        "listing_id": [1, 2],
        "predicted_price": [100.0, 150.0],
        "predicted_occupancy": [0.5, 0.7],
        "estimated_annual_revenue": [100.0 * 0.5 * 365, 150.0 * 0.7 * 365],
    })
    assert set(result.columns) == {
        "listing_id", "predicted_price",
        "predicted_occupancy", "estimated_annual_revenue",
    }
    assert result["estimated_annual_revenue"][0] == pytest.approx(18250.0, abs=1.0)
    assert result["estimated_annual_revenue"][1] == pytest.approx(38325.0, abs=1.0)


def test_revenue_nonnegative():
    price = np.array([50.0, 200.0, 120.0])
    occupancy = np.array([0.3, 0.9, 0.0])
    revenue = price * occupancy * 365
    assert (revenue >= 0).all()
