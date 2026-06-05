import pickle
from pathlib import Path

import numpy as np
import polars as pl

from geoai.config import DB_PATH
from geoai.models.features import build_feature_matrix, prepare_X_y_price, prepare_X_y_occupancy
from geoai.models.price import MODEL_PATH as PRICE_MODEL_PATH
from geoai.models.occupancy import MODEL_PATH as OCCUPANCY_MODEL_PATH


def estimate_revenue(db_path: Path = DB_PATH) -> pl.DataFrame:
    df = build_feature_matrix(db_path)
    with open(PRICE_MODEL_PATH, "rb") as f:
        price_artifact = pickle.load(f)
    with open(OCCUPANCY_MODEL_PATH, "rb") as f:
        occ_artifact = pickle.load(f)
    X_price, _, _ = prepare_X_y_price(df)
    X_occ, _, _ = prepare_X_y_occupancy(df)
    predicted_price = np.exp(price_artifact["model"].predict(X_price))
    predicted_occupancy = occ_artifact["model"].predict(X_occ).clip(0, 1)
    estimated_annual_revenue = predicted_price * predicted_occupancy * 365
    return pl.DataFrame({
        "listing_id": df["id"].to_list(),
        "predicted_price": predicted_price.tolist(),
        "predicted_occupancy": predicted_occupancy.tolist(),
        "estimated_annual_revenue": estimated_annual_revenue.tolist(),
    })
