import duckdb
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from geoai.api.deps import get_db
from geoai.api.schemas import StatsResponse

router = APIRouter()


@router.get("/api/stats", response_model=StatsResponse)
def get_stats(db: duckdb.DuckDBPyConnection = Depends(get_db)):
    row = db.execute("""
        SELECT
            AVG(predicted_price)       AS avg_price,
            AVG(predicted_occupancy)   AS avg_occupancy,
            MEDIAN(predicted_revenue)  AS median_revenue,
            COUNT(*)                   AS listing_count
        FROM listing_predictions
    """).fetchone()
    content = StatsResponse(
        avg_price=row[0],
        avg_occupancy=row[1],
        median_revenue=row[2],
        listing_count=row[3],
    ).model_dump()
    return JSONResponse(content=content, headers={"Cache-Control": "max-age=3600"})
