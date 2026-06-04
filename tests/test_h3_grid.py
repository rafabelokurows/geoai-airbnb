import polars as pl
import pytest

from geoai.features.h3_grid import assign_h3_cells, compute_hex_aggregates


def _listings() -> pl.DataFrame:
    return pl.DataFrame({
        "id": [1, 2, 3, 4],
        "latitude":  [41.14961, 41.14970, 41.15200, 41.20000],
        "longitude": [-8.61099, -8.61105, -8.61300, -8.65000],
        "price": [100.0, 120.0, 150.0, 200.0],
        "walkability_score": [75.0, 80.0, 60.0, 50.0],
        "occupancy_rate_365d": [0.6, 0.7, 0.5, 0.4],
    })


def test_assign_h3_cells_returns_listing_id_and_h3_cell():
    result = assign_h3_cells(_listings())
    assert "listing_id" in result.columns
    assert "h3_cell_r8" in result.columns


def test_assign_h3_cells_all_assigned():
    result = assign_h3_cells(_listings())
    assert result["h3_cell_r8"].null_count() == 0


def test_assign_h3_cells_same_location_same_cell():
    listings = pl.DataFrame({
        "id": [1, 2],
        "latitude":  [41.14961, 41.14961],
        "longitude": [-8.61099, -8.61099],
        "price": [100.0, 200.0],
        "walkability_score": [70.0, 80.0],
        "occupancy_rate_365d": [0.5, 0.6],
    })
    result = assign_h3_cells(listings)
    assert result["h3_cell_r8"][0] == result["h3_cell_r8"][1]


def test_compute_hex_aggregates_listing_count():
    df = pl.DataFrame({
        "listing_id": [1, 2, 3],
        "h3_cell_r8": ["cell_A", "cell_A", "cell_B"],
        "price": [100.0, 200.0, 150.0],
        "walkability_score": [70.0, 80.0, 60.0],
        "occupancy_rate_365d": [0.5, 0.6, 0.4],
    })
    result = compute_hex_aggregates(df)
    cell_a = result.filter(pl.col("h3_cell") == "cell_A")
    assert cell_a["listing_count"][0] == 2
    assert cell_a["avg_price"][0] == pytest.approx(150.0, abs=0.1)
