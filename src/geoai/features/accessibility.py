from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from geoai.config import DB_PATH, PORTO_CENTER_LAT, PORTO_CENTER_LON, PORTO_LANDMARKS
from geoai.database.warehouse import init_warehouse


def haversine_km(
    lats: np.ndarray, lons: np.ndarray,
    target_lat: float, target_lon: float,
) -> np.ndarray:
    R = 6371.0
    dlat = np.radians(target_lat - lats)
    dlon = np.radians(target_lon - lons)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lats)) * np.cos(np.radians(target_lat)) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(a))


def _nearest_km(
    listing_lats: np.ndarray,
    listing_lons: np.ndarray,
    poi_lats: np.ndarray,
    poi_lons: np.ndarray,
) -> list[float | None]:
    if len(poi_lats) == 0:
        return [None] * len(listing_lats)
    results = []
    for lat, lon in zip(listing_lats, listing_lons):
        dists = haversine_km(poi_lats, poi_lons, lat, lon)
        results.append(float(np.min(dists)))
    return results


def compute_accessibility(
    listings: pl.DataFrame,
    pois: pl.DataFrame,
) -> pl.DataFrame:
    lats = listings["latitude"].to_numpy()
    lons = listings["longitude"].to_numpy()

    metro = pois.filter(
        (pl.col("poi_type") == "railway") & (pl.col("poi_subtype") == "subway_entrance")
    )
    stations = pois.filter(
        (pl.col("poi_type") == "railway") & (pl.col("poi_subtype") == "station")
    )
    supermarkets = pois.filter(pl.col("poi_subtype") == "supermarket")

    city_center_dists = haversine_km(lats, lons, PORTO_CENTER_LAT, PORTO_CENTER_LON)
    metro_dists = _nearest_km(lats, lons, metro["latitude"].to_numpy(), metro["longitude"].to_numpy())
    station_dists = _nearest_km(lats, lons, stations["latitude"].to_numpy(), stations["longitude"].to_numpy())
    supermarket_dists = _nearest_km(lats, lons, supermarkets["latitude"].to_numpy(), supermarkets["longitude"].to_numpy())

    new_cols = [
        pl.Series("dist_city_center_km", city_center_dists.tolist()),
        pl.Series("dist_nearest_metro_km", metro_dists),
        pl.Series("dist_nearest_station_km", station_dists),
        pl.Series("dist_nearest_supermarket_km", supermarket_dists),
    ]
    for name, (lat, lon) in PORTO_LANDMARKS.items():
        dists = haversine_km(lats, lons, lat, lon)
        new_cols.append(pl.Series(f"dist_{name}_km", dists.tolist()))

    return listings.select("id").with_columns(new_cols)


def load_accessibility_into_db(db_path: Path = DB_PATH) -> int:
    _con = init_warehouse(db_path)
    _con.close()
    with duckdb.connect(str(db_path)) as con:
        listings = pl.from_arrow(
            con.execute("SELECT id, latitude, longitude FROM listings").arrow()
        )
        pois = pl.from_arrow(
            con.execute(
                "SELECT osm_id, poi_type, poi_subtype, latitude, longitude FROM poi_features"
                " WHERE poi_type = 'railway' OR poi_subtype = 'supermarket'"
            ).arrow()
        )
    result = compute_accessibility(listings, pois)
    feature_cols = [c for c in result.columns if c != "id"]
    col_list = "listing_id, " + ", ".join(feature_cols)
    placeholders = ", ".join(["?"] * (len(feature_cols) + 1))
    update_set = ", ".join(f"{c} = excluded.{c}" for c in feature_cols)
    with duckdb.connect(str(db_path)) as con:
        for row in result.iter_rows(named=True):
            values = [row["id"]] + [row[c] for c in feature_cols]
            con.execute(
                f"""
                INSERT INTO listing_features ({col_list})
                VALUES ({placeholders})
                ON CONFLICT (listing_id) DO UPDATE SET {update_set}
                """,
                values,
            )
    with duckdb.connect(str(db_path)) as con:
        return con.execute("SELECT COUNT(*) FROM listing_features").fetchone()[0]
