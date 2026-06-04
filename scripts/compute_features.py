import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geoai.features.runner import run_all_features

if __name__ == "__main__":
    print("=== GeoAI Phase 2: Feature Engineering ===\n")
    summary = run_all_features()
    print("\n=== Summary ===")
    print(f"Calendar rows loaded  : {summary['calendar_rows']:>10,}")
    print(f"Listings with features: {summary['listings_with_features']:>10,}")
    print(f"H3 cells              : {summary['h3_cells']:>10,}")
    print("\nPhase 2 complete.")
