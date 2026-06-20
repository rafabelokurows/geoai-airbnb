import polars as pl
from fastapi import APIRouter, Depends

from geoai.api.deps import AppState, get_state
from geoai.api.schemas import NeighbourhoodRank

router = APIRouter()


@router.get("/neighbourhoods", response_model=list[NeighbourhoodRank])
def neighbourhoods(state: AppState = Depends(get_state)) -> list[NeighbourhoodRank]:
    df = (
        state.listings_df
        .filter(
            pl.col("neighbourhood").is_not_null()
            & pl.col("estimated_annual_revenue").is_not_null()
        )
        .group_by("neighbourhood")
        .agg([
            pl.col("estimated_annual_revenue").mean().alias("avg_revenue"),
            pl.col("id").count().alias("listing_count"),
        ])
        .sort("avg_revenue", descending=True)
    )
    return [
        NeighbourhoodRank(
            neighbourhood=row["neighbourhood"],
            listing_count=int(row["listing_count"]),
            avg_revenue=float(row["avg_revenue"]),
        )
        for row in df.iter_rows(named=True)
    ]
