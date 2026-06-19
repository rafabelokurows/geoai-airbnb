import httpx
import polars as pl
import duckdb
from pathlib import Path
from urllib.parse import urlparse

from geoai.config import RAW_AIRBNB_DIR, DB_PATH, AIRBNB_PORTO_URL
from geoai.database.warehouse import init_warehouse

_KEEP_COLS = [
    "id", "name", "description", "latitude", "longitude", "neighbourhood_cleansed",
    "room_type", "property_type", "accommodates", "bedrooms", "beds",
    "price", "minimum_nights", "maximum_nights",
    "availability_30", "availability_60", "availability_90", "availability_365",
    "number_of_reviews", "review_scores_rating", "reviews_per_month",
    "host_id", "host_name", "host_is_superhost", "amenities", "last_scraped",
]


def _download_raw(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name
    dest_path = dest_dir / filename
    if dest_path.exists():
        return dest_path
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=8192):
                f.write(chunk)
    return dest_path


def _read_raw(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=10000, ignore_errors=True)


def clean_listings(df: pl.DataFrame) -> pl.DataFrame:
    # Parse price: "$1,234.00" → 1234.0
    df = df.with_columns(
        pl.col("price")
        .str.replace_all(r"[\$,]", "")
        .cast(pl.Float64, strict=False)
        .alias("price")
    )
    # Parse superhost: "t"/"f" → bool
    df = df.with_columns(
        pl.when(pl.col("host_is_superhost") == "t").then(pl.lit(True))
        .when(pl.col("host_is_superhost") == "f").then(pl.lit(False))
        .otherwise(pl.lit(None, dtype=pl.Boolean))
        .alias("host_is_superhost")
    )
    # Select available columns from the desired set
    existing = [c for c in _KEEP_COLS if c in df.columns]
    df = df.select(existing)
    # Rename neighbourhood column
    if "neighbourhood_cleansed" in df.columns:
        df = df.rename({"neighbourhood_cleansed": "neighbourhood"})
    # Drop rows missing coordinates or id (unusable for geospatial work)
    return df.drop_nulls(subset=["id", "latitude", "longitude"])


def load_airbnb_into_db(
    db_path: Path = DB_PATH,
    url: str = AIRBNB_PORTO_URL,
) -> int:
    # Ensure warehouse schema exists
    _con = init_warehouse(db_path)
    _con.close()

    raw_path = _download_raw(url, RAW_AIRBNB_DIR)
    df = _read_raw(raw_path)
    df = clean_listings(df)

    with duckdb.connect(str(db_path)) as con:
        table_cols = [row[0] for row in con.execute("DESCRIBE listings").fetchall()]
        df_cols_available = [c for c in table_cols if c in df.columns]
        df = df.select(df_cols_available)

        col_list = ", ".join(df_cols_available)
        con.execute("BEGIN")
        con.execute("DELETE FROM listings")
        con.execute(f"INSERT INTO listings ({col_list}) SELECT {col_list} FROM df")
        con.execute("COMMIT")
        return con.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
