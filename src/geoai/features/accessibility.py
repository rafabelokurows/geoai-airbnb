from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from geoai.config import DB_PATH, PORTO_CENTER_LAT, PORTO_CENTER_LON
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

    city_center_dists = haversine_km(lats, lons, PORTO_CENTER_LAT, PORTO_CENTER_LON)
    metro_dists = _nearest_km(lats, lons, metro["latitude"].to_numpy(), metro["longitude"].to_numpy())
    station_dists = _nearest_km(lats, lons, stations["latitude"].to_numpy(), stations["longitude"].to_numpy())

    return listings.select("id").with_columns([
        pl.Series("dist_city_center_km", city_center_dists.tolist()),
        pl.Series("dist_nearest_metro_km", metro_dists),
        pl.Series("dist_nearest_station_km", station_dists),
    ])


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
                " WHERE poi_type = 'railway'"
            ).arrow()
        )
    result = compute_accessibility(listings, pois)
    with duckdb.connect(str(db_path)) as con:
        for row in result.iter_rows(named=True):
            con.execute(
                """
                INSERT INTO listing_features (listing_id, dist_city_center_km,
                    dist_nearest_metro_km, dist_nearest_station_km)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (listing_id) DO UPDATE SET
                    dist_city_center_km = excluded.dist_city_center_km,
                    dist_nearest_metro_km = excluded.dist_nearest_metro_km,
                    dist_nearest_station_km = excluded.dist_nearest_station_km
                """,
                [row["id"], row["dist_city_center_km"],
                 row["dist_nearest_metro_km"], row["dist_nearest_station_km"]],
            )
    with duckdb.connect(str(db_path)) as con:
        return con.execute("SELECT COUNT(*) FROM listing_features").fetchone()[0]
