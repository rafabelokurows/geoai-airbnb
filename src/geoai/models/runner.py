import argparse
from pathlib import Path

from geoai.config import DB_PATH
from geoai.models.evaluate import run_evaluation
from geoai.models.predict import run_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GeoAI ML models and print evaluation report")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Path to DuckDB warehouse")
    args = parser.parse_args()

    metrics = run_evaluation(args.db)

    print("\n=== Evaluation Summary ===")
    print(f"Price RMSE:     €{metrics['price_rmse']:.2f}  {'✓' if metrics['price_target_met'] else '✗'} target <€60")
    print(f"Occupancy MAE:   {metrics['occupancy_mae']:.4f}  {'✓' if metrics['occupancy_target_met'] else '✗'} target <0.15")
    print(f"Median Revenue:  €{metrics['median_annual_revenue']:,.0f}/year")

    print("\nComputing predictions and SHAP values...")
    run_predictions(args.db)
    print("Done. Start the API with: uvicorn geoai.api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
