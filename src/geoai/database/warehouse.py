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
    amenities            VARCHAR,
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


def init_warehouse(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(_CREATE_LISTINGS)
    con.execute(_CREATE_POI_FEATURES)
    return con


def get_connection(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path))
