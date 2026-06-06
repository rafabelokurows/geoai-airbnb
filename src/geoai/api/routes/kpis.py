import polars as pl
from fastapi import APIRouter, Depends

from geoai.api.deps import AppState, get_state
from geoai.api.schemas import KpiResponse

router = APIRouter()


@router.get("/kpis", response_model=KpiResponse)
def kpis(state: AppState = Depends(get_state)) -> KpiResponse:
    df = state.listings_df
    with_preds = df.filter(pl.col("predicted_price").is_not_null())
    return KpiResponse(
        listing_count=len(df),
        listings_with_predictions=len(with_preds),
        avg_price=float(df["price"].drop_nulls().mean()),
        avg_occupancy=float(with_preds["predicted_occupancy"].mean()),
        median_annual_revenue=float(with_preds["estimated_annual_revenue"].median()),
    )
