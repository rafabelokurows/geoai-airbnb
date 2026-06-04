"""Run full Phase 1 ingestion for Porto. Requires network access for OSM."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from geoai.database.warehouse import init_warehouse, get_connection
from geoai.ingestion.airbnb import load_airbnb_into_db
from geoai.ingestion.osm import load_osm_into_db
from geoai.config import DB_PATH


def main():
    print(f"Initializing warehouse at {DB_PATH}")
    con = init_warehouse()
    con.close()

    print("Ingesting Airbnb listings...")
    n_listings = load_airbnb_into_db()
    print(f"  Loaded {n_listings:,} listings")

    print("Ingesting OSM POIs for Porto (network request)...")
    n_pois = load_osm_into_db()
    print(f"  Loaded {n_pois:,} POIs")

    print("\nDone. Warehouse summary:")
    con = get_connection()
    print(con.execute("SELECT COUNT(*) as listings FROM listings").fetchdf().to_string(index=False))
    print(con.execute(
        "SELECT poi_type, COUNT(*) as count FROM poi_features GROUP BY poi_type ORDER BY count DESC"
    ).fetchdf().to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
