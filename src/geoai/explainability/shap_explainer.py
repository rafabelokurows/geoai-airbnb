import pickle
from pathlib import Path

import numpy as np
import polars as pl
import shap

from geoai.config import DB_PATH
from geoai.models.features import build_feature_matrix, prepare_X_y_price, prepare_X_y_occupancy
from geoai.models.price import MODEL_PATH as PRICE_MODEL_PATH
from geoai.models.occupancy import MODEL_PATH as OCCUPANCY_MODEL_PATH


def build_explainer(model, X: np.ndarray) -> shap.TreeExplainer:
    return shap.TreeExplainer(model, data=shap.sample(X, min(100, len(X))))


def compute_global_importance(
    explainer: shap.TreeExplainer,
    X: np.ndarray,
    feature_names: list[str],
) -> pl.DataFrame:
    shap_values = explainer.shap_values(X)
    importance = np.abs(shap_values).mean(axis=0)
    df = pl.DataFrame({"feature": feature_names, "importance": importance.tolist()})
    return df.sort("importance", descending=True)


def explain_listing(
    explainer: shap.TreeExplainer,
    x_row: np.ndarray,
    feature_names: list[str],
    top_n: int | None = None,
) -> dict:
    row = x_row.reshape(1, -1)
    shap_values = explainer.shap_values(row)[0]
    base_value = float(explainer.expected_value)
    predicted_value = float(base_value + shap_values.sum())

    drivers = sorted(
        [{"feature": f, "impact": float(v)} for f, v in zip(feature_names, shap_values)],
        key=lambda d: abs(d["impact"]),
        reverse=True,
    )
    if top_n is not None:
        drivers = drivers[:top_n]

    return {
        "predicted_value": predicted_value,
        "base_value": base_value,
        "drivers": drivers,
    }


def run_shap_analysis(db_path: Path = DB_PATH) -> dict:
    df = build_feature_matrix(db_path)

    with open(PRICE_MODEL_PATH, "rb") as f:
        price_artifact = pickle.load(f)
    with open(OCCUPANCY_MODEL_PATH, "rb") as f:
        occ_artifact = pickle.load(f)

    X_price, _, price_features = prepare_X_y_price(df)
    X_occ, _, occ_features = prepare_X_y_occupancy(df)

    price_explainer = build_explainer(price_artifact["model"], X_price)
    occ_explainer = build_explainer(occ_artifact["model"], X_occ)

    price_importance = compute_global_importance(price_explainer, X_price, price_features)
    occ_importance = compute_global_importance(occ_explainer, X_occ, occ_features)

    return {
        "price_explainer": price_explainer,
        "occupancy_explainer": occ_explainer,
        "price_importance": price_importance,
        "occupancy_importance": occ_importance,
        "price_feature_names": price_features,
        "occupancy_feature_names": occ_features,
        "X_price": X_price,
        "X_occ": X_occ,
        "listing_ids": df["id"].to_list(),
    }
