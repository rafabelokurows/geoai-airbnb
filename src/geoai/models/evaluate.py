from pathlib import Path

from geoai.config import DB_PATH
from geoai.models.price import train_price_model
from geoai.models.occupancy import train_occupancy_model
from geoai.models.revenue import estimate_revenue


def run_evaluation(db_path: Path = DB_PATH) -> dict:
    print("Training price model...")
    price = train_price_model(db_path)
    for group, m in price.get("per_type", {}).items():
        print(f"  [{group}] RMSE: EUR{m['rmse']:.2f}  (n_train={m['n_train']}, n_test={m['n_test']})")

    print("Training occupancy model...")
    occ = train_occupancy_model(db_path)
    for group, m in occ.get("per_type", {}).items():
        print(f"  [{group}] MAE: {m['mae']:.4f}  (n_train={m['n_train']}, n_test={m['n_test']})")

    revenue_df = estimate_revenue(db_path)
    median_revenue = float(revenue_df["estimated_annual_revenue"].median())
    print(f"  Median estimated annual revenue: EUR{median_revenue:,.0f}")

    return {
        "price_rmse": price["rmse"],
        "occupancy_mae": occ["mae"],
        "median_annual_revenue": median_revenue,
        "price_target_met": price["rmse"] < 60.0,
        "occupancy_target_met": occ["mae"] < 0.15,
    }
