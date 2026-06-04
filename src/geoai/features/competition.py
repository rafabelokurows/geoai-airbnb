from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from geoai.config import DB_PATH
from geoai.database.warehouse import init_warehouse
from geoai.features.accessibility import haversine_km

_KM_PER_DEG = 111.0


def compute_competition(listings: pl.DataFrame) -> pl.DataFrame:
    lats = listings["latitude"].to_numpy()
    lons = listings["longitude"].to_numpy()
    prices = listings["price"].to_numpy(allow_copy=True).astype(float)
    ids = listings["id"].to_list()
    neighbourhoods = listings["neighbourhood"].to_list()

    # neighbourhood medians — computed once
    neighbourhood_medians: dict[str, float] = {}
    for nbhd in set(neighbourhoods):
        mask = np.array([n == nbhd for n in neighbourhoods])
        nbhd_prices = prices[mask]
        valid = nbhd_prices[~np.isnan(nbhd_prices)]
        if len(valid) > 0:
            neighbourhood_medians[nbhd] = float(np.median(valid))

    rows = []
    for i, (listing_id, lat, lon, nbhd) in enumerate(zip(ids, lats, lons, neighbourhoods)):
        # bounding box pre-filter for 1km (largest radius needed)
        deg = 1.0 / _KM_PER_DEG
        mask_bbox = (
            (np.abs(lats - lat) <= deg)
            & (np.abs(lons - lon) <= deg)
        )
        mask_bbox[i] = False  # exclude self

        nearby_lats = lats[mask_bbox]
        nearby_lons = lons[mask_bbox]
        nearby_prices = prices[mask_bbox]

        if len(nearby_lats) == 0:
            rows.append({
                "listing_id": listing_id,
                "listings_500m": 0,
                "listings_1km": 0,
                "avg_price_500m": None,
                "median_price_neighbourhood": neighbourhood_medians.get(nbhd),
            })
            continue

        dists = haversine_km(nearby_lats, nearby_lons, lat, lon)
        mask_500 = dists <= 0.5
        mask_1km = dists <= 1.0
        prices_500 = nearby_prices[mask_500]
        valid_500 = prices_500[~np.isnan(prices_500)]

        rows.append({
            "listing_id": listing_id,
            "listings_500m": int(np.sum(mask_500)),
            "listings_1km": int(np.sum(mask_1km)),
            "avg_price_500m": float(np.mean(valid_500)) if len(valid_500) > 0 else None,
            "median_price_neighbourhood": neighbourhood_medians.get(nbhd),
        })
    return pl.DataFrame(rows)


def load_competition_into_db(db_path: Path = DB_PATH) -> int:
    _con = init_warehouse(db_path)
    _con.close()
    with duckdb.connect(str(db_path)) as con:
        listings = pl.from_arrow(
            con.execute(
                "SELECT id, latitude, longitude, price, neighbourhood FROM listings"
            ).arrow()
        )
    result = compute_competition(listings)
    cols = [c for c in result.columns if c != "listing_id"]
    update_set = ", ".join(f"{c} = excluded.{c}" for c in cols)
    col_list = ", ".join(result.columns)
    with duckdb.connect(str(db_path)) as con:
        con.execute(f"""
            INSERT INTO listing_features ({col_list})
            SELECT {col_list} FROM result
            ON CONFLICT (listing_id) DO UPDATE SET {update_set}
        """)
        return con.execute("SELECT COUNT(*) FROM listing_features").fetchone()[0]
