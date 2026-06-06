from fastapi import APIRouter, Depends, Query

from geoai.api.deps import AppState, get_state
from geoai.api.schemas import ListingPoint, ListingsResponse

router = APIRouter()


@router.get("/listings", response_model=ListingsResponse)
def listings(
    limit: int = Query(default=5000, le=15000),
    offset: int = Query(default=0, ge=0),
    state: AppState = Depends(get_state),
) -> ListingsResponse:
    df = state.listings_df
    total = len(df)
    page = df.slice(offset, limit)
    return ListingsResponse(
        listings=[
            ListingPoint(
                id=int(row["id"]),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                price=float(row["price"]) if row["price"] is not None else None,
                room_type=row["room_type"],
                predicted_price=float(row["predicted_price"]) if row["predicted_price"] is not None else None,
                predicted_occupancy=float(row["predicted_occupancy"]) if row["predicted_occupancy"] is not None else None,
                estimated_annual_revenue=float(row["estimated_annual_revenue"]) if row["estimated_annual_revenue"] is not None else None,
            )
            for row in page.iter_rows(named=True)
        ],
        total=total,
    )
