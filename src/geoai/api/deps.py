import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import shap

from geoai.config import DB_PATH
from geoai.explainability.shap_explainer import build_explainer
from geoai.models.features import build_feature_matrix, prepare_X_y_occupancy, prepare_X_y_price
from geoai.models.occupancy import MODEL_PATH as OCCUPANCY_MODEL_PATH
from geoai.models.price import MODEL_PATH as PRICE_MODEL_PATH


@dataclass
class AppState:
    listings_df: pl.DataFrame = None
    hex_df: pl.DataFrame = None
    price_explainer: shap.TreeExplainer = None
    occ_explainer: shap.TreeExplainer = None
    X_price: np.ndarray = None
    price_features: list[str] = field(default_factory=list)
    listing_id_to_idx: dict = field(default_factory=dict)


app_state = AppState()


def load_app_state(db_path: Path = DB_PATH) -> None:
    import duckdb

    # Feature matrix + predictions
    df = build_feature_matrix(db_path)
    X_price, _, price_features = prepare_X_y_price(df)
    X_occ, _, _ = prepare_X_y_occupancy(df)

    with open(PRICE_MODEL_PATH, "rb") as f:
        price_artifact = pickle.load(f)
    with open(OCCUPANCY_MODEL_PATH, "rb") as f:
        occ_artifact = pickle.load(f)

    predicted_price = np.exp(price_artifact["model"].predict(X_price))
    predicted_occupancy = occ_artifact["model"].predict(X_occ).clip(0, 1)
    estimated_revenue = predicted_price * predicted_occupancy * 365

    revenue_df = pl.DataFrame({
        "listing_id": df["id"].to_list(),
        "predicted_price": predicted_price.tolist(),
        "predicted_occupancy": predicted_occupancy.tolist(),
        "estimated_annual_revenue": estimated_revenue.tolist(),
    })

    # Raw listings with geo
    with duckdb.connect(str(db_path), read_only=True) as con:
        listings_raw = pl.from_arrow(con.execute("""
            SELECT l.id, l.latitude, l.longitude, l.price, l.room_type,
                   l.neighbourhood, lf.h3_cell_r8
            FROM listings l
            LEFT JOIN listing_features lf ON l.id = lf.listing_id
        """).arrow())
        hex_raw = pl.from_arrow(con.execute(
            "SELECT * FROM hex_aggregates"
        ).arrow())

    listings_df = listings_raw.join(revenue_df, left_on="id", right_on="listing_id", how="left")

    # Add avg_revenue per hex
    hex_revenue = (
        listings_df
        .filter(pl.col("h3_cell_r8").is_not_null())
        .group_by("h3_cell_r8")
        .agg(pl.col("estimated_annual_revenue").mean().alias("avg_revenue"))
    )
    hex_df = hex_raw.join(hex_revenue, on="h3_cell_r8", how="left")

    # SHAP
    price_explainer = build_explainer(price_artifact["model"], X_price)
    listing_id_to_idx = {int(lid): i for i, lid in enumerate(df["id"].to_list())}

    app_state.listings_df = listings_df
    app_state.hex_df = hex_df
    app_state.price_explainer = price_explainer
    app_state.X_price = X_price
    app_state.price_features = list(price_features)
    app_state.listing_id_to_idx = listing_id_to_idx


def get_state() -> AppState:
    return app_state
