import polars as pl
from fastapi import APIRouter, Depends, Query

from geoai.api.deps import AppState, get_state
from geoai.api.schemas import PriceGapListing, PriceGapResponse, SegmentRow

router = APIRouter()

_UNDER_THRESHOLD = 0.85
_OVER_THRESHOLD = 1.15


def _price_gap_df(listings_df: pl.DataFrame) -> pl.DataFrame:
    return (
        listings_df
        .filter(
            pl.col("predicted_price").is_not_null()
            & pl.col("price").is_not_null()
            & (pl.col("predicted_price") > 0)
            & pl.col("latitude").is_not_null()
            & pl.col("longitude").is_not_null()
        )
        .with_columns(
            ((pl.col("price") - pl.col("predicted_price")) / pl.col("predicted_price"))
            .alias("price_gap_pct"),
            (pl.col("predicted_price") - pl.col("price")).alias("opportunity_gap"),
        )
        .with_columns(
            pl.when(pl.col("price") < pl.col("predicted_price") * _UNDER_THRESHOLD)
            .then(pl.lit("underpriced"))
            .when(pl.col("price") > pl.col("predicted_price") * _OVER_THRESHOLD)
            .then(pl.lit("overpriced"))
            .otherwise(pl.lit("fair"))
            .alias("direction"),
            (pl.col("opportunity_gap") * pl.col("predicted_occupancy") * 365)
            .alias("estimated_uplift_annual"),
        )
    )


def _to_listing(row: dict) -> PriceGapListing:
    return PriceGapListing(
        listing_id=str(int(row["id"])),
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        actual_price=float(row["price"]),
        predicted_price=float(row["predicted_price"]),
        price_gap_pct=float(row["price_gap_pct"]),
        opportunity_gap=float(row["opportunity_gap"]),
        estimated_uplift_annual=float(row["estimated_uplift_annual"]),
        room_type=row.get("room_type"),
        neighbourhood=row.get("neighbourhood"),
        direction=row["direction"],
    )


@router.get("/price-gap", response_model=PriceGapResponse)
def price_gap(
    top_n: int = Query(default=50, le=200),
    state: AppState = Depends(get_state),
) -> PriceGapResponse:
    df = _price_gap_df(state.listings_df)

    underpriced = (
        df.filter(pl.col("direction") == "underpriced")
        .sort("opportunity_gap", descending=True)
        .head(top_n)
    )
    overpriced = (
        df.filter(pl.col("direction") == "overpriced")
        .sort("price_gap_pct", descending=True)
        .head(top_n)
    )

    segments: list[SegmentRow] = []
    for seg_type, seg_col in [("room_type", "room_type"), ("neighbourhood", "neighbourhood")]:
        seg_df = (
            df.filter(pl.col(seg_col).is_not_null())
            .group_by(seg_col)
            .agg([
                pl.len().alias("total_listings"),
                (pl.col("direction") == "underpriced").sum().alias("underpriced_count"),
                (pl.col("direction") == "overpriced").sum().alias("overpriced_count"),
                (pl.col("direction") == "fair").sum().alias("fair_count"),
                pl.col("price_gap_pct").median().alias("median_gap_pct"),
            ])
            .sort("total_listings", descending=True)
        )
        for row in seg_df.iter_rows(named=True):
            segments.append(SegmentRow(
                segment_type=seg_type,
                segment_value=str(row[seg_col]),
                total_listings=int(row["total_listings"]),
                underpriced_count=int(row["underpriced_count"]),
                overpriced_count=int(row["overpriced_count"]),
                fair_count=int(row["fair_count"]),
                median_gap_pct=float(row["median_gap_pct"]),
            ))

    return PriceGapResponse(
        underpriced=[_to_listing(r) for r in underpriced.iter_rows(named=True)],
        overpriced=[_to_listing(r) for r in overpriced.iter_rows(named=True)],
        segment_summary=segments,
    )
