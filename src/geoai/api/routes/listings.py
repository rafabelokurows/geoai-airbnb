import duckdb
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from geoai.api.deps import get_db
from geoai.api.schemas import ListingPoint

router = APIRouter()


@router.get("/api/listings", response_model=list[ListingPoint])
def get_listings(
    hex_id: str,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    rows = db.execute("""
        SELECT listing_id, latitude, longitude, predicted_price, predicted_occupancy
        FROM listing_predictions
        WHERE h3_cell_r8 = ?
        ORDER BY listing_id
    """, [hex_id]).fetchall()
    content = [
        ListingPoint(
            id=r[0], latitude=r[1], longitude=r[2],
            predicted_price=r[3], predicted_occupancy=r[4],
        ).model_dump()
        for r in rows
    ]
    return JSONResponse(content=content, headers={"Cache-Control": "max-age=3600"})
