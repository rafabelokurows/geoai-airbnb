from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from geoai.config import DB_PATH
from geoai.database.warehouse import init_warehouse
from geoai.features.accessibility import haversine_km

_KM_PER_DEG = 111.0  # approximate degrees latitude per km


def count_within_radius(
    listing_lat: float,
    listing_lon: float,
    pois: pl.DataFrame,
    radius_km: float,
) -> int:
    if len(pois) == 0:
        return 0
    # bounding box pre-filter — cheap arithmetic, cuts candidates before trig
    deg = radius_km / _KM_PER_DEG
    nearby = pois.filter(
        (pl.col("latitude").is_between(listing_lat - deg, listing_lat + deg))
        & (pl.col("longitude").is_between(listing_lon - deg, listing_lon + deg))
    )
    if len(nearby) == 0:
        return 0
    lats = nearby["latitude"].to_numpy()
    lons = nearby["longitude"].to_numpy()
    dists = haversine_km(lats, lons, listing_lat, listing_lon)
    return int(np.sum(dists <= radius_km))


_RESTAURANT_SUBTYPES = ["restaurant", "fast_food"]
_BAR_SUBTYPES = ["bar", "pub"]
_CAFE_SUBTYPES = ["cafe"]
_SUPERMARKET_SUBTYPES = ["supermarket"]
_ATTRACTION_SUBTYPES = ["attraction", "gallery", "viewpoint", "museum", "theatre", "cinema"]
_MUSEUM_SUBTYPES = ["museum"]
_PARK_SUBTYPES = ["park", "garden"]


def compute_poi_density(
    listings: pl.DataFrame,
    pois: pl.DataFrame,
) -> pl.DataFrame:
    restaurants  = pois.filter(pl.col("poi_subtype").is_in(_RESTAURANT_SUBTYPES))
    bars         = pois.filter(pl.col("poi_subtype").is_in(_BAR_SUBTYPES))
    cafes        = pois.filter(pl.col("poi_subtype").is_in(_CAFE_SUBTYPES))
    supermarkets = pois.filter(pl.col("poi_subtype").is_in(_SUPERMARKET_SUBTYPES))
    attractions  = pois.filter(pl.col("poi_subtype").is_in(_ATTRACTION_SUBTYPES))
    museums      = pois.filter(pl.col("poi_subtype").is_in(_MUSEUM_SUBTYPES))
    parks        = pois.filter(pl.col("poi_subtype").is_in(_PARK_SUBTYPES))

    rows = []
    for row in listings.iter_rows(named=True):
        lat, lon = row["latitude"], row["longitude"]
        rows.append({
            "listing_id":       row["id"],
            "restaurants_250m": count_within_radius(lat, lon, restaurants,  0.25),
            "restaurants_500m": count_within_radius(lat, lon, restaurants,  0.5),
            "bars_500m":        count_within_radius(lat, lon, bars,         0.5),
            "cafes_500m":       count_within_radius(lat, lon, cafes,        0.5),
            "supermarkets_1km": count_within_radius(lat, lon, supermarkets, 1.0),
            "attractions_1km":  count_within_radius(lat, lon, attractions,  1.0),
            "museums_2km":      count_within_radius(lat, lon, museums,      2.0),
            "parks_500m":       count_within_radius(lat, lon, parks,        0.5),
        })
    return pl.DataFrame(rows)


def load_poi_density_into_db(db_path: Path = DB_PATH) -> int:
    _con = init_warehouse(db_path)
    _con.close()
    with duckdb.connect(str(db_path)) as con:
        listings = pl.from_arrow(
            con.execute("SELECT id, latitude, longitude FROM listings").arrow()
        )
        pois = pl.from_arrow(
            con.execute(
                "SELECT osm_id, poi_type, poi_subtype, latitude, longitude FROM poi_features"
            ).arrow()
        )
    result = compute_poi_density(listings, pois)
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
