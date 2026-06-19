import polars as pl
from fastapi import APIRouter, Depends, Query

from geoai.api.deps import AppState, get_state
from geoai.api.schemas import OpportunityListing

router = APIRouter()


@router.get("/opportunities", response_model=list[OpportunityListing])
def opportunities(
    top_n: int = Query(default=50, le=500),
    state: AppState = Depends(get_state),
) -> list[OpportunityListing]:
    df = (
        state.listings_df
        .filter(
            pl.col("predicted_price").is_not_null()
            & pl.col("price").is_not_null()
            & pl.col("latitude").is_not_null()
            & pl.col("longitude").is_not_null()
        )
        .with_columns(
            (pl.col("predicted_price") - pl.col("price")).alias("opportunity_gap"),
        )
        .filter(pl.col("price") < pl.col("predicted_price") * 0.85)
        .with_columns(
            (pl.col("opportunity_gap") * pl.col("predicted_occupancy") * 365)
            .alias("estimated_uplift_annual")
        )
        .sort("opportunity_gap", descending=True)
        .head(top_n)
    )
    return [
        OpportunityListing(
            listing_id=str(int(row["id"])),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            actual_price=float(row["price"]),
            predicted_price=float(row["predicted_price"]),
            opportunity_gap=float(row["opportunity_gap"]),
            estimated_uplift_annual=float(row["estimated_uplift_annual"]),
        )
        for row in df.iter_rows(named=True)
    ]
