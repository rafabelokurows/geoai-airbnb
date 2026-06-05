from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from geoai.config import DB_PATH

NUMERIC_FEATURE_COLS = [
    "accommodates", "bedrooms", "beds", "minimum_nights",
    "review_scores_rating", "host_is_superhost", "number_of_reviews",
    "dist_city_center_km", "dist_nearest_metro_km", "dist_nearest_station_km",
    "dist_nearest_supermarket_km", "dist_airport_km", "travel_time_airport_min",
    "restaurants_250m", "restaurants_500m", "bars_500m", "cafes_500m",
    "supermarkets_1km", "attractions_1km", "museums_2km", "parks_500m",
    "amenity_density_1km", "restaurant_density",
    "listings_500m", "listings_1km", "avg_price_500m", "median_price_neighbourhood",
    "walkability_score",
]


def build_feature_matrix(db_path: Path = DB_PATH) -> pl.DataFrame:
    with duckdb.connect(str(db_path)) as con:
        return pl.from_arrow(con.execute("""
            SELECT
                l.id,
                l.accommodates,
                l.bedrooms,
                l.beds,
                l.room_type,
                l.minimum_nights,
                l.review_scores_rating,
                CAST(l.host_is_superhost AS INTEGER) AS host_is_superhost,
                l.number_of_reviews,
                l.price AS target_price,
                lf.dist_city_center_km,
                lf.dist_nearest_metro_km,
                lf.dist_nearest_station_km,
                lf.dist_nearest_supermarket_km,
                lf.dist_airport_km,
                lf.travel_time_airport_min,
                lf.restaurants_250m,
                lf.restaurants_500m,
                lf.bars_500m,
                lf.cafes_500m,
                lf.supermarkets_1km,
                lf.attractions_1km,
                lf.museums_2km,
                lf.parks_500m,
                lf.amenity_density_1km,
                lf.restaurant_density,
                lf.listings_500m,
                lf.listings_1km,
                lf.avg_price_500m,
                lf.median_price_neighbourhood,
                lf.walkability_score,
                lf.occupancy_rate_365d AS target_occupancy
            FROM listings l
            JOIN listing_features lf ON l.id = lf.listing_id
            WHERE l.price IS NOT NULL
              AND l.price > 0
              AND lf.occupancy_rate_365d IS NOT NULL
        """).arrow())


def prepare_X_y_price(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    room_dummies = df.select("room_type").to_dummies()
    numeric = df.select(NUMERIC_FEATURE_COLS).fill_null(0)
    X_df = pl.concat([numeric, room_dummies], how="horizontal")
    X = X_df.to_numpy().astype(np.float32)
    y = np.log(df["target_price"].to_numpy().astype(np.float64))
    return X, y, X_df.columns


def prepare_X_y_occupancy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    room_dummies = df.select("room_type").to_dummies()
    price_col = df.select(pl.col("target_price").alias("price"))
    numeric = df.select(NUMERIC_FEATURE_COLS).fill_null(0)
    X_df = pl.concat([numeric, price_col, room_dummies], how="horizontal")
    X = X_df.to_numpy().astype(np.float32)
    y = df["target_occupancy"].to_numpy().astype(np.float64)
    return X, y, X_df.columns
