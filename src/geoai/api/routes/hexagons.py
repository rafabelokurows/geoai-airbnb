from typing import Literal

import duckdb
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from geoai.api.deps import get_db
from geoai.api.schemas import HexDetail, HexSummary

router = APIRouter()

_MODE_COLUMN = {
    "price": "avg_price",
    "occupancy": "avg_occupancy",
    "revenue": "avg_revenue",
}


@router.get("/api/hexagons", response_model=list[HexSummary])
def list_hexagons(
    mode: Literal["price", "occupancy", "revenue"] = "price",
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    col = _MODE_COLUMN[mode]
    rows = db.execute(f"""
        SELECT h3_cell_r8 AS hex_id, {col} AS value, listing_count
        FROM hex_aggregates
        ORDER BY hex_id
    """).fetchall()
    content = [HexSummary(hex_id=r[0], value=r[1], listing_count=r[2]).model_dump() for r in rows]
    return JSONResponse(content=content, headers={"Cache-Control": "max-age=3600"})


@router.get("/api/hexagons/{hex_id}", response_model=HexDetail)
def get_hex_detail(
    hex_id: str,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    row = db.execute("""
        SELECT h3_cell_r8, avg_price, avg_occupancy, avg_revenue,
               listing_count, avg_walkability_score, avg_restaurant_density,
               avg_dist_city_center_km, avg_competition_score
        FROM hex_aggregates
        WHERE h3_cell_r8 = ?
    """, [hex_id]).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="hex not found")

    content = HexDetail(
        hex_id=row[0], avg_price=row[1], avg_occupancy=row[2],
        avg_revenue=row[3], listing_count=row[4],
        avg_walkability_score=row[5], avg_restaurant_density=row[6],
        avg_dist_city_center_km=row[7], avg_competition_score=row[8],
    ).model_dump()
    return JSONResponse(content=content, headers={"Cache-Control": "max-age=3600"})
