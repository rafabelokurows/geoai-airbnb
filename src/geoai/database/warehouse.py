import duckdb
from pathlib import Path

from geoai.config import DB_PATH

_CREATE_LISTINGS = """
CREATE TABLE IF NOT EXISTS listings (
    id                   BIGINT PRIMARY KEY,
    name                 VARCHAR,
    latitude             DOUBLE,
    longitude            DOUBLE,
    neighbourhood        VARCHAR,
    room_type            VARCHAR,
    property_type        VARCHAR,
    accommodates         INTEGER,
    bedrooms             DOUBLE,
    beds                 DOUBLE,
    price                DOUBLE,
    minimum_nights       INTEGER,
    maximum_nights       INTEGER,
    availability_30      INTEGER,
    availability_60      INTEGER,
    availability_90      INTEGER,
    availability_365     INTEGER,
    number_of_reviews    INTEGER,
    review_scores_rating DOUBLE,
    reviews_per_month    DOUBLE,
    host_id              BIGINT,
    host_name            VARCHAR,
    host_is_superhost    BOOLEAN,
    amenities            JSON,
    last_scraped         DATE
)
"""

_CREATE_POI_FEATURES = """
CREATE TABLE IF NOT EXISTS poi_features (
    osm_id       VARCHAR PRIMARY KEY,
    poi_type     VARCHAR,
    poi_subtype  VARCHAR,
    name         VARCHAR,
    latitude     DOUBLE,
    longitude    DOUBLE,
    geometry_wkt VARCHAR
)
"""


_CREATE_CALENDAR = """
CREATE TABLE IF NOT EXISTS calendar (
    listing_id      BIGINT,
    date            DATE,
    available       BOOLEAN,
    price           DOUBLE,
    minimum_nights  INTEGER,
    maximum_nights  INTEGER,
    PRIMARY KEY (listing_id, date)
)
"""

_CREATE_LISTING_FEATURES = """
CREATE TABLE IF NOT EXISTS listing_features (
    listing_id                  BIGINT PRIMARY KEY,
    dist_city_center_km         DOUBLE,
    dist_nearest_metro_km       DOUBLE,
    dist_nearest_station_km     DOUBLE,
    dist_nearest_supermarket_km DOUBLE,
    dist_livraria_lello_km      DOUBLE,
    dist_torre_clerigos_km      DOUBLE,
    dist_ribeira_km             DOUBLE,
    dist_ponte_luis_km          DOUBLE,
    dist_mercado_bolhao_km      DOUBLE,
    dist_jardins_cristal_km     DOUBLE,
    restaurants_250m            INTEGER,
    restaurants_500m            INTEGER,
    bars_500m                   INTEGER,
    cafes_500m                  INTEGER,
    supermarkets_1km            INTEGER,
    attractions_1km             INTEGER,
    museums_2km                 INTEGER,
    parks_500m                  INTEGER,
    listings_500m               INTEGER,
    listings_1km                INTEGER,
    avg_price_500m              DOUBLE,
    median_price_neighbourhood  DOUBLE,
    walkability_score           DOUBLE,
    dist_airport_km             DOUBLE,
    travel_time_airport_min     DOUBLE,
    amenity_density_1km         INTEGER,
    restaurant_density          DOUBLE,
    h3_cell_r8                  VARCHAR,
    occupancy_rate_30d          DOUBLE,
    occupancy_rate_90d          DOUBLE,
    occupancy_rate_365d         DOUBLE
)
"""

_CREATE_HEX_AGGREGATES = """
CREATE TABLE IF NOT EXISTS hex_aggregates (
    h3_cell         VARCHAR PRIMARY KEY,
    listing_count   INTEGER,
    avg_price       DOUBLE,
    avg_occupancy   DOUBLE,
    avg_walkability DOUBLE
)
"""


def init_warehouse(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(_CREATE_LISTINGS)
    con.execute(_CREATE_POI_FEATURES)
    con.execute(_CREATE_CALENDAR)
    con.execute(_CREATE_LISTING_FEATURES)
    con.execute(_CREATE_HEX_AGGREGATES)
    return con


def get_connection(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Return a live connection to the warehouse. Caller must close it."""
    return duckdb.connect(str(db_path))
