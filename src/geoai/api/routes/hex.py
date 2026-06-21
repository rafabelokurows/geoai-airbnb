import polars as pl
from fastapi import APIRouter, Depends

from geoai.api.deps import AppState, get_state
from geoai.api.routes.price_gap import _price_gap_df, _to_listing
from geoai.api.schemas import HexCell, HexListing, PriceGapResponse

router = APIRouter()


@router.get("/hex-aggregates", response_model=list[HexCell])
def hex_aggregates(state: AppState = Depends(get_state)) -> list[HexCell]:
    df = state.hex_df.drop_nulls(subset=["h3_cell_r8"])
    return [
        HexCell(
            h3_cell=row["h3_cell_r8"],
            listing_count=int(row["listing_count"]),
            avg_price=float(row["avg_price"] or 0),
            avg_occupancy=float(row["avg_occupancy"] or 0),
            avg_revenue=float(row["avg_revenue"] or 0),
        )
        for row in df.iter_rows(named=True)
    ]


@router.get("/hex/{h3_cell}/price-gap", response_model=PriceGapResponse)
def hex_price_gap(h3_cell: str, state: AppState = Depends(get_state)) -> PriceGapResponse:
    hex_listings_df = state.listings_df.filter(pl.col("h3_cell_r8") == h3_cell)
    df = _price_gap_df(hex_listings_df)
    underpriced = (
        df.filter(pl.col("direction") == "underpriced")
        .sort("opportunity_gap", descending=True)
        .head(10)
    )
    overpriced = (
        df.filter(pl.col("direction") == "overpriced")
        .sort("price_gap_pct", descending=True)
        .head(10)
    )
    return PriceGapResponse(
        underpriced=[_to_listing(r) for r in underpriced.iter_rows(named=True)],
        overpriced=[_to_listing(r) for r in overpriced.iter_rows(named=True)],
        segment_summary=[],
    )


@router.get("/hex/{h3_cell}/listings", response_model=list[HexListing])
def hex_listings(h3_cell: str, state: AppState = Depends(get_state)) -> list[HexListing]:
    df = state.listings_df.filter(pl.col("h3_cell_r8") == h3_cell)
    return [
        HexListing(
            price=float(row["price"]) if row["price"] is not None else None,
            predicted_occupancy=float(row["predicted_occupancy"]) if row["predicted_occupancy"] is not None else None,
        )
        for row in df.iter_rows(named=True)
    ]
