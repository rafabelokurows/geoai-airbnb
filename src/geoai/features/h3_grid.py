from pathlib import Path

import duckdb
import h3
import polars as pl

from geoai.config import DB_PATH
from geoai.database.warehouse import init_warehouse

H3_RESOLUTION = 8


def assign_h3_cells(listings: pl.DataFrame) -> pl.DataFrame:
    cells = [
        h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
        for lat, lon in zip(listings["latitude"].to_list(), listings["longitude"].to_list())
    ]
    return pl.DataFrame({
        "listing_id": listings["id"].to_list(),
        "h3_cell_r8": cells,
    })


def compute_hex_aggregates(df: pl.DataFrame) -> pl.DataFrame:
    # df must have: listing_id, h3_cell_r8, price, walkability_score, occupancy_rate_365d
    return (
        df.group_by("h3_cell_r8")
        .agg([
            pl.len().alias("listing_count"),
            pl.col("price").mean().alias("avg_price"),
            pl.col("occupancy_rate_365d").mean().alias("avg_occupancy"),
            pl.col("walkability_score").mean().alias("avg_walkability"),
        ])
        .rename({"h3_cell_r8": "h3_cell"})
    )


def load_h3_into_db(db_path: Path = DB_PATH) -> tuple[int, int]:
    _con = init_warehouse(db_path)
    _con.close()
    with duckdb.connect(str(db_path)) as con:
        listings = pl.from_arrow(
            con.execute("SELECT id, latitude, longitude FROM listings").arrow()
        )
        features = pl.from_arrow(
            con.execute(
                "SELECT listing_id, price, walkability_score, occupancy_rate_365d"
                " FROM listing_features"
            ).arrow()
        )

    cells = assign_h3_cells(listings)

    with duckdb.connect(str(db_path)) as con:
        con.execute("""
            INSERT INTO listing_features (listing_id, h3_cell_r8)
            SELECT listing_id, h3_cell_r8 FROM cells
            ON CONFLICT (listing_id) DO UPDATE SET h3_cell_r8 = excluded.h3_cell_r8
        """)

    joined = features.join(cells, on="listing_id")
    hex_df = compute_hex_aggregates(joined)

    col_list = ", ".join(hex_df.columns)
    cols_no_pk = [c for c in hex_df.columns if c != "h3_cell"]
    update_set = ", ".join(f"{c} = excluded.{c}" for c in cols_no_pk)
    with duckdb.connect(str(db_path)) as con:
        con.execute(f"""
            INSERT INTO hex_aggregates ({col_list})
            SELECT {col_list} FROM hex_df
            ON CONFLICT (h3_cell) DO UPDATE SET {update_set}
        """)
        n_listings = con.execute("SELECT COUNT(*) FROM listing_features WHERE h3_cell_r8 IS NOT NULL").fetchone()[0]
        n_hexes = con.execute("SELECT COUNT(*) FROM hex_aggregates").fetchone()[0]
    return n_listings, n_hexes
