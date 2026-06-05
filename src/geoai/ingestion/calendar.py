from pathlib import Path

import duckdb
import polars as pl

from geoai.config import CALENDAR_URL, DB_PATH, RAW_AIRBNB_DIR
from geoai.database.warehouse import init_warehouse
from geoai.ingestion.airbnb import _download_raw

_KEEP_COLS = [
    "listing_id", "date", "available", "price", "minimum_nights", "maximum_nights"
]


def clean_calendar(df: pl.DataFrame) -> pl.DataFrame:
    transforms = [
        pl.when(pl.col("available") == "t").then(pl.lit(True))
        .when(pl.col("available") == "f").then(pl.lit(False))
        .otherwise(pl.lit(None, dtype=pl.Boolean))
        .alias("available"),
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("date"),
    ]
    if "price" in df.columns:
        transforms.append(
            pl.col("price").str.replace_all(r"[\$,]", "").cast(pl.Float64, strict=False).alias("price")
        )
    df = df.with_columns(transforms)
    existing = [c for c in _KEEP_COLS if c in df.columns]
    return df.select(existing).drop_nulls(subset=["listing_id", "date"])


def load_calendar_into_db(
    db_path: Path = DB_PATH, url: str = CALENDAR_URL
) -> int:
    _con = init_warehouse(db_path)
    _con.close()
    raw_path = _download_raw(url, RAW_AIRBNB_DIR)
    df = pl.read_csv(raw_path, infer_schema_length=10000)
    df = clean_calendar(df)
    col_list = ", ".join(df.columns)
    with duckdb.connect(str(db_path)) as con:
        con.execute("BEGIN")
        con.execute("DELETE FROM calendar")
        con.execute(f"INSERT INTO calendar ({col_list}) SELECT {col_list} FROM df")
        con.execute("COMMIT")
        return con.execute("SELECT COUNT(*) FROM calendar").fetchone()[0]
