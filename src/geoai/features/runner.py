from pathlib import Path

from geoai.config import DB_PATH
from geoai.ingestion.calendar import load_calendar_into_db
from geoai.features.accessibility import load_accessibility_into_db
from geoai.features.poi_density import load_poi_density_into_db
from geoai.features.competition import load_competition_into_db
from geoai.features.walkability import load_walkability_into_db
from geoai.features.h3_grid import load_h3_into_db
from geoai.features.occupancy import load_occupancy_into_db


def run_all_features(db_path: Path = DB_PATH) -> dict:
    print("Step 1/7: Calendar ingestion...")
    n_calendar = load_calendar_into_db(db_path)
    print(f"  {n_calendar:,} calendar rows loaded")

    print("Step 2/7: Accessibility features...")
    n_acc = load_accessibility_into_db(db_path)
    print(f"  {n_acc:,} listings with accessibility features")

    print("Step 3/7: POI density features...")
    n_poi = load_poi_density_into_db(db_path)
    print(f"  {n_poi:,} listings with POI density features")

    print("Step 4/7: Competition features...")
    n_comp = load_competition_into_db(db_path)
    print(f"  {n_comp:,} listings with competition features")

    print("Step 5/7: Walkability score...")
    n_walk = load_walkability_into_db(db_path)
    print(f"  {n_walk:,} listings with walkability score")

    print("Step 6/7: Occupancy estimation...")
    n_occ = load_occupancy_into_db(db_path)
    print(f"  {n_occ:,} listings with occupancy rates")

    print("Step 7/7: H3 hexagon assignment...")
    n_listings_h3, n_hexes = load_h3_into_db(db_path)
    print(f"  {n_listings_h3:,} listings assigned to {n_hexes:,} H3 cells")

    return {
        "calendar_rows": n_calendar,
        "listings_with_features": n_listings_h3,
        "h3_cells": n_hexes,
    }
