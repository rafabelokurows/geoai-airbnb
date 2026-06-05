from typing import Literal

import duckdb
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from geoai.api.deps import get_db
from geoai.api.schemas import HexShapResponse, ShapDriver, ShapFeature

router = APIRouter()


@router.get("/api/shap/global", response_model=list[ShapFeature])
def get_shap_global(
    model: Literal["price", "occupancy"] = "price",
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    rows = db.execute("""
        SELECT feature, importance
        FROM shap_global
        WHERE model = ?
        ORDER BY importance DESC
    """, [model]).fetchall()
    content = [ShapFeature(feature=r[0], importance=r[1]).model_dump() for r in rows]
    return JSONResponse(content=content, headers={"Cache-Control": "max-age=3600"})


@router.get("/api/shap/{hex_id}", response_model=HexShapResponse)
def get_shap_hex(
    hex_id: str,
    model: Literal["price", "occupancy"] = "price",
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    rows = db.execute("""
        SELECT feature, avg_impact, base_value
        FROM hex_shap
        WHERE h3_cell_r8 = ? AND model = ?
        ORDER BY ABS(avg_impact) DESC
    """, [hex_id, model]).fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="hex not found")

    base_value = rows[0][2]
    drivers = [ShapDriver(feature=r[0], avg_impact=r[1]).model_dump() for r in rows]
    content = HexShapResponse(
        hex_id=hex_id,
        base_value=base_value,
        drivers=drivers,
    ).model_dump()
    return JSONResponse(content=content, headers={"Cache-Control": "max-age=3600"})
