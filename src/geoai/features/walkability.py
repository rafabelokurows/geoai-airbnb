from pathlib import Path

import duckdb
import polars as pl

from geoai.config import DB_PATH
from geoai.database.warehouse import init_warehouse


def _cap_norm(value, cap: float) -> float:
    if value is None or (isinstance(value, float) and value != value):
        return 0.0
    return min(float(value), cap) / cap


def _proximity_score(dist_km, max_km: float) -> float:
    if dist_km is None or (isinstance(dist_km, float) and dist_km != dist_km):
        return 0.0
    return 1.0 - min(float(dist_km), max_km) / max_km


def compute_walkability(features: pl.DataFrame) -> pl.DataFrame:
    scores = []
    for row in features.iter_rows(named=True):
        restaurant_score = _cap_norm(row.get("restaurants_500m"), 10)
        bar_cafe_score   = _cap_norm((row.get("bars_500m") or 0) + (row.get("cafes_500m") or 0), 10)
        transit_score    = _proximity_score(row.get("dist_nearest_metro_km"), 1.0)
        center_score     = _proximity_score(row.get("dist_city_center_km"), 3.0)
        park_score       = _cap_norm(row.get("parks_500m"), 3)
        supermarket_score = _cap_norm(row.get("supermarkets_1km"), 2)

        raw = (
            0.20 * restaurant_score
            + 0.10 * bar_cafe_score
            + 0.30 * transit_score
            + 0.20 * center_score
            + 0.10 * park_score
            + 0.10 * supermarket_score
        )
        scores.append(max(0.0, min(100.0, raw * 100)))

    return features.select("listing_id").with_columns(
        pl.Series("walkability_score", scores)
    )


def load_walkability_into_db(db_path: Path = DB_PATH) -> int:
    _con = init_warehouse(db_path)
    _con.close()
    with duckdb.connect(str(db_path)) as con:
        features = pl.from_arrow(con.execute("""
            SELECT listing_id, restaurants_500m, bars_500m, cafes_500m,
                   dist_nearest_metro_km, dist_city_center_km,
                   parks_500m, supermarkets_1km
            FROM listing_features
        """).arrow())
    result = compute_walkability(features)
    with duckdb.connect(str(db_path)) as con:
        con.execute("""
            INSERT INTO listing_features (listing_id, walkability_score)
            SELECT listing_id, walkability_score FROM result
            ON CONFLICT (listing_id) DO UPDATE SET walkability_score = excluded.walkability_score
        """)
        return con.execute("SELECT COUNT(*) FROM listing_features").fetchone()[0]
