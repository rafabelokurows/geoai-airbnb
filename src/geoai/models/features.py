import json
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from geoai.config import DB_PATH

# Each key becomes a binary feature column; value is the set of raw amenity strings
# that count as a match (case-insensitive substring).
AMENITY_GROUPS: dict[str, list[str]] = {
    "amenity_kitchen": ["Kitchen", "Kitchenette"],
    "amenity_heating": ["Heating", "Central heating", "Heating - split type ductless system"],
    "amenity_ac": [
        "Central air conditioning",
        "AC - split type ductless system",
        "Air conditioning",
        "Window AC unit",
    ],
    "amenity_parking": [
        "Free parking garage on premises",
        "Free residential garage on premises",
        "Free parking on premises",
        "Free street parking",
        "Paid parking on premises",
        "Paid parking off premises",
    ],
    "amenity_beach": [
        "Beach access",
        "Shared beach access",
        "Beach access – Beachfront",
        "Shared beach access – Beachfront",
    ],
    "amenity_safety": [
        "Fire extinguisher",
        "First aid kit",
        "Smoke alarm",
        "Carbon monoxide alarm",
    ],
    "amenity_views": [
        "Beach view",
        "Ocean View",
        "Garden view",
        "City skyline view",
        "Mountain view",
        "Courtyard view",
    ],
    "amenity_luxury": [
        "Pool",
        "Private pool",
        "Hot tub",
        "Sauna",
        "Gym",
        "Private gym",
        "Exercise room",
        "Exercise equipment"
    ],
    "amenity_pool": [
        "Pool",
        "Private pool",
        "Shared outdoor pool",
    ],
    "amenity_backyard": ["Private backyard", "Private backyard – Not fully fenced"],
    "amenity_outdoor": [
        "Private patio or balcony",
        "Private backyard",
        "Private backyard – Not fully fenced",
        "Outdoor dining area",
        "Outdoor furniture",
        "BBQ grill",
        "Garden",
    ],
    "amenity_family": [
        "Crib",
        "High chair",
        "Children's books and toys",
        "Pack ’n play/Travel crib",
        "Baby bath",
    ],
    "amenity_water_view": ["Beach view", "Ocean View"],
    "amenity_crib": ["Crib"],
    "amenity_private_entrance": ["Private entrance"],
    "amenity_elevator": ["Elevator"],
    "amenity_patio_balcony": ["Private patio or balcony"],
    "amenity_breakfast": ["Breakfast"],
    "amenity_entertainment": [
        "TV",
        "Cable TV",
        "HDTV",
        "Netflix",
        "Amazon Prime Video",
        "Disney+",
        "Sound system",
        "Movie theater",
        "Game console",
        "Arcade"
    ],
    "premium_brands": [
        "Bosch",
        "Smeg",
        "Bose",
        "Nintendo Switch",
        "PS4",
        "PS5",
        "Rituals",
        "Aroma de Portugal",
        "Aromas de Portugal",
        "Castelbel",
        "Marshall"
    ],
}

AMENITY_COLS = list(AMENITY_GROUPS.keys())

# Binary flags from listing name + description (case-insensitive substring match)
DESC_KEYWORD_GROUPS: dict[str, list[str]] = {
    "kw_luxury":    ["luxury", "luxurious", "upscale"],
    "kw_boutique":  ["boutique"],
    "kw_villa":     ["villa"],
    "kw_pool":      ["pool"],
    "kw_jacuzzi":   ["jacuzzi", "hot tub", "whirlpool"],
    "kw_exclusive": ["exclusive"],
    "kw_penthouse": ["penthouse"],
    "kw_panoramic": ["panoramic", "rooftop", "stunning view"],
}

DESC_KW_COLS = list(DESC_KEYWORD_GROUPS.keys())

NUMERIC_FEATURE_COLS = [
    "accommodates", "bedrooms", "beds",
    "review_scores_rating", "host_is_superhost", "number_of_reviews",
    "dist_city_center_km", "dist_nearest_metro_km", "dist_nearest_station_km",
    "dist_nearest_supermarket_km", "dist_airport_km", "travel_time_airport_min",
    "restaurants_250m", "restaurants_500m", "bars_500m", "cafes_500m",
    "supermarkets_1km", "attractions_1km", "museums_2km", "parks_500m",
    "amenity_density_1km", "restaurant_density",
    "listings_500m", "listings_1km", "avg_price_500m", "median_price_neighbourhood",
    "walkability_score",
] + AMENITY_COLS + DESC_KW_COLS


def _desc_flags(name_series: pl.Series, desc_series: pl.Series) -> pl.DataFrame:
    rows = []
    for name, desc in zip(name_series, desc_series):
        text = ((name or "") + " " + (desc or "")).lower()
        row = {
            col: int(any(kw in text for kw in kws))
            for col, kws in DESC_KEYWORD_GROUPS.items()
        }
        rows.append(row)
    return pl.DataFrame(rows, schema={c: pl.Int8 for c in DESC_KW_COLS})


def _parse_amenities(raw: str | None) -> set[str]:
    if not raw:
        return set()
    try:
        return set(json.loads(raw))
    except Exception:
        return set()


def _amenity_flags(amenities_series: pl.Series) -> pl.DataFrame:
    rows = []
    for raw in amenities_series:
        present = _parse_amenities(raw)
        row = {}
        for col, keywords in AMENITY_GROUPS.items():
            row[col] = int(any(
                any(kw.lower() in item.lower() for kw in keywords)
                for item in present
            ))
        rows.append(row)
    return pl.DataFrame(rows, schema={c: pl.Int8 for c in AMENITY_COLS})


def build_feature_matrix(db_path: Path = DB_PATH) -> pl.DataFrame:
    with duckdb.connect(str(db_path)) as con:
        base = pl.from_arrow(con.execute("""
            SELECT
                l.id,
                l.name,
                l.description,
                l.accommodates,
                l.bedrooms,
                l.beds,
                l.room_type,
                l.minimum_nights,
                l.review_scores_rating,
                CAST(l.host_is_superhost AS INTEGER) AS host_is_superhost,
                l.number_of_reviews,
                l.price AS target_price,
                l.amenities,
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

    amenity_flags = _amenity_flags(base["amenities"])
    desc_flags = _desc_flags(base["name"], base["description"])
    return pl.concat([base.drop(["amenities", "name", "description"]), amenity_flags, desc_flags], how="horizontal")


_ENTIRE_HOME = "Entire home/apt"


def split_by_room_type(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Two groups: 'Entire home/apt' and 'other' (all remaining room types)."""
    entire = df.filter(pl.col("room_type") == _ENTIRE_HOME)
    other = df.filter(pl.col("room_type") != _ENTIRE_HOME)
    groups = {}
    if len(entire) > 0:
        groups[_ENTIRE_HOME] = entire
    if len(other) > 0:
        groups["other"] = other
    return groups


def prepare_X_y_price(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_df = df.select(NUMERIC_FEATURE_COLS).fill_null(0)
    X = X_df.to_numpy().astype(np.float32)
    y = np.log(df["target_price"].to_numpy().astype(np.float64))
    return X, y, X_df.columns


def prepare_X_y_occupancy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    price_col = df.select(pl.col("target_price").alias("price"))
    numeric = df.select(NUMERIC_FEATURE_COLS).fill_null(0)
    X_df = pl.concat([numeric, price_col], how="horizontal")
    X = X_df.to_numpy().astype(np.float32)
    y = df["target_occupancy"].to_numpy().astype(np.float64)
    return X, y, X_df.columns
