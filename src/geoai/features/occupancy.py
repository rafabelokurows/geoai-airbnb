import datetime
from pathlib import Path

import duckdb
import polars as pl

from geoai.config import DB_PATH
from geoai.database.warehouse import init_warehouse


def compute_occupancy(calendar: pl.DataFrame) -> pl.DataFrame:
    max_date = calendar["date"].max()
    if max_date is None:
        return pl.DataFrame(schema={
            "listing_id": pl.Int64,
            "occupancy_rate_30d": pl.Float64,
            "occupancy_rate_90d": pl.Float64,
            "occupancy_rate_365d": pl.Float64,
        })

    windows = {
        "occupancy_rate_30d": 30,
        "occupancy_rate_90d": 90,
        "occupancy_rate_365d": 365,
    }

    result = calendar.select("listing_id").unique()

    for col_name, days in windows.items():
        cutoff = max_date - datetime.timedelta(days=days)
        window_df = (
            calendar.filter(pl.col("date") > cutoff)
            .group_by("listing_id")
            .agg([
                (pl.col("available") == False).sum().alias("booked"),  # noqa: E712
                pl.len().alias("total"),
            ])
            .with_columns(
                (pl.col("booked").cast(pl.Float64) / pl.col("total")).alias(col_name)
            )
            .select("listing_id", col_name)
        )
        result = result.join(window_df, on="listing_id", how="left")

    return result


def load_occupancy_into_db(db_path: Path = DB_PATH) -> int:
    _con = init_warehouse(db_path)
    _con.close()
    with duckdb.connect(str(db_path)) as con:
        calendar = pl.from_arrow(
            con.execute("SELECT listing_id, date, available FROM calendar").arrow()
        )
    result = compute_occupancy(calendar)
    if len(result) == 0:
        return 0
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
