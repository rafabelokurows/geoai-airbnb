import osmnx as ox
import geopandas as gpd
import polars as pl
import duckdb
from pathlib import Path
from typing import Any

from geoai.config import OSM_CITY, OSM_POI_TAGS, DB_PATH
from geoai.database.warehouse import init_warehouse

_TAG_PRIORITY = ["amenity", "tourism", "leisure", "railway"]


def _fetch_osm(city: str, tags: dict[str, Any]) -> gpd.GeoDataFrame:
    return ox.features_from_place(city, tags=tags)


def process_pois(gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    records = []
    for idx, row in gdf.iterrows():
        element_type, osm_id = idx[0], idx[1]
        uid = f"{element_type}_{osm_id}"

        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        centroid = geom if geom.geom_type == "Point" else geom.centroid

        poi_type = None
        poi_subtype = None
        for tag_key in _TAG_PRIORITY:
            val = row.get(tag_key)
            if val and str(val) not in ("nan", "None", ""):
                poi_type = tag_key
                poi_subtype = str(val)
                break

        if poi_type is None:
            continue

        name_val = row.get("name")
        records.append({
            "osm_id": uid,
            "poi_type": poi_type,
            "poi_subtype": poi_subtype,
            "name": str(name_val) if name_val and str(name_val) not in ("nan", "None") else None,
            "latitude": centroid.y,
            "longitude": centroid.x,
            "geometry_wkt": geom.wkt,
        })

    return pl.DataFrame(
        records,
        schema={
            "osm_id": pl.Utf8,
            "poi_type": pl.Utf8,
            "poi_subtype": pl.Utf8,
            "name": pl.Utf8,
            "latitude": pl.Float64,
            "longitude": pl.Float64,
            "geometry_wkt": pl.Utf8,
        },
    )


def load_osm_into_db(
    db_path: Path = DB_PATH,
    city: str = OSM_CITY,
    tags: dict = OSM_POI_TAGS,
) -> int:
    # Ensure warehouse schema exists
    _con = init_warehouse(db_path)
    _con.close()

    gdf = _fetch_osm(city, tags)
    df = process_pois(gdf)

    col_list = ", ".join(df.columns)
    with duckdb.connect(str(db_path)) as con:
        con.execute("BEGIN")
        con.execute("DELETE FROM poi_features")
        con.execute(f"INSERT INTO poi_features ({col_list}) SELECT {col_list} FROM df")
        con.execute("COMMIT")
        return con.execute("SELECT COUNT(*) FROM poi_features").fetchone()[0]
