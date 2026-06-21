import pickle
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import shap as shap_lib

from geoai.config import DB_PATH
from geoai.models.features import (
    _ENTIRE_HOME,
    build_feature_matrix,
    prepare_X_y_price,
    prepare_X_y_occupancy,
)
from geoai.models.price import MODEL_PATH as PRICE_MODEL_PATH
from geoai.models.occupancy import MODEL_PATH as OCCUPANCY_MODEL_PATH


def _load_model(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. "
            "Run `python -m geoai.models.runner` first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _room_group(room_type: str) -> str:
    return room_type if room_type == _ENTIRE_HOME else "other"


def _route_predict(models_dict: dict, X_all: np.ndarray, room_types: list[str]) -> np.ndarray:
    """Route each row to its group model; no SHAP."""
    fallback = _ENTIRE_HOME if _ENTIRE_HOME in models_dict else next(iter(models_dict))
    preds = np.zeros(len(room_types))
    for group in {_room_group(rt) for rt in room_types}:
        model = models_dict.get(group, models_dict[fallback])
        mask = np.array([_room_group(r) == group for r in room_types])
        preds[mask] = model.predict(X_all[mask])
    return preds


def _predict_and_shap(
    models_dict: dict,
    X_all: np.ndarray,
    room_types: list[str],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Route rows to per-group models; return (predictions, shap_vals, weighted_base)."""
    n, n_feat = X_all.shape
    preds = np.zeros(n)
    shap_vals = np.zeros((n, n_feat))
    base_sum = 0.0

    fallback = _ENTIRE_HOME if _ENTIRE_HOME in models_dict else next(iter(models_dict))
    groups = {_room_group(rt) for rt in room_types}

    for group in groups:
        model = models_dict.get(group, models_dict[fallback])
        mask = np.array([_room_group(r) == group for r in room_types])
        X_sub = X_all[mask]
        preds[mask] = model.predict(X_sub)
        explainer = shap_lib.TreeExplainer(model, data=shap_lib.sample(X_sub, min(100, len(X_sub))))
        shap_vals[mask] = explainer.shap_values(X_sub)
        base_sum += float(explainer.expected_value) * mask.sum()

    return preds, shap_vals, base_sum / n


def _build_hex_shap_df(
    shap_vals: np.ndarray,
    features: list[str],
    model_name: str,
    base_val: float,
    h3_cells: list[str],
) -> pl.DataFrame:
    df = pl.DataFrame({"h3_cell_r8": h3_cells, **{f: shap_vals[:, j].tolist() for j, f in enumerate(features)}})
    melted = df.unpivot(index="h3_cell_r8", variable_name="feature", value_name="impact")
    return (
        melted.group_by(["h3_cell_r8", "feature"])
        .agg(pl.col("impact").mean().alias("avg_impact"))
        .with_columns(pl.lit(model_name).alias("model"), pl.lit(base_val).alias("base_value"))
    )


def run_predictions(db_path: Path = DB_PATH) -> None:
    price_artifact = _load_model(PRICE_MODEL_PATH)
    occ_artifact = _load_model(OCCUPANCY_MODEL_PATH)

    df = build_feature_matrix(db_path)
    X_price, _, price_features = prepare_X_y_price(df)
    X_occ, _, occ_features = prepare_X_y_occupancy(df)

    room_types = df["room_type"].to_list()

    price_raw, price_shap_vals, price_base = _predict_and_shap(price_artifact["models"], X_price, room_types)
    predicted_price = np.exp(price_raw).astype(np.float64)

    occ_raw, occ_shap_vals, occ_base = _predict_and_shap(occ_artifact["models"], X_occ, room_types)
    predicted_occupancy = occ_raw.clip(0.0, 1.0).astype(np.float64)

    predicted_revenue = predicted_price * predicted_occupancy * 30.0
    listing_ids = df["id"].to_list()

    with duckdb.connect(str(db_path)) as con:
        meta = pl.from_arrow(con.execute("""
            SELECT lf.listing_id, lf.h3_cell_r8,
                   l.latitude, l.longitude,
                   lf.walkability_score, lf.restaurant_density,
                   lf.dist_city_center_km, lf.listings_500m
            FROM listing_features lf
            JOIN listings l ON l.id = lf.listing_id
        """).arrow())

    id_order = {v: i for i, v in enumerate(listing_ids)}
    meta = (
        meta.filter(pl.col("listing_id").is_in(listing_ids))
        .with_columns(
            pl.Series("_order", [id_order[i] for i in meta.filter(pl.col("listing_id").is_in(listing_ids))["listing_id"].to_list()])
        )
        .sort("_order")
        .drop("_order")
    )

    preds_df = pl.DataFrame({
        "listing_id": listing_ids,
        "h3_cell_r8": meta["h3_cell_r8"].to_list(),
        "predicted_price": predicted_price.tolist(),
        "predicted_occupancy": predicted_occupancy.tolist(),
        "predicted_revenue": predicted_revenue.tolist(),
        "latitude": meta["latitude"].to_list(),
        "longitude": meta["longitude"].to_list(),
    })

    price_imp = np.abs(price_shap_vals).mean(axis=0)
    occ_imp = np.abs(occ_shap_vals).mean(axis=0)
    shap_global_df = pl.concat([
        pl.DataFrame({"model": ["price"] * len(price_features), "feature": list(price_features), "importance": price_imp.tolist()}),
        pl.DataFrame({"model": ["occupancy"] * len(occ_features), "feature": list(occ_features), "importance": occ_imp.tolist()}),
    ])

    h3_cells = preds_df["h3_cell_r8"].to_list()
    hex_shap_df = pl.concat([
        _build_hex_shap_df(price_shap_vals, list(price_features), "price", price_base, h3_cells),
        _build_hex_shap_df(occ_shap_vals, list(occ_features), "occupancy", occ_base, h3_cells),
    ])

    with duckdb.connect(str(db_path)) as con:
        preds_arrow = preds_df.to_arrow()
        con.register("_preds", preds_arrow)
        con.execute("CREATE OR REPLACE TABLE listing_predictions AS SELECT * FROM _preds")
        con.unregister("_preds")

        con.execute("""
            CREATE OR REPLACE TABLE hex_aggregates AS
            SELECT
                p.h3_cell_r8,
                COUNT(*) AS listing_count,
                AVG(p.predicted_price)       AS avg_price,
                AVG(p.predicted_occupancy)   AS avg_occupancy,
                AVG(p.predicted_revenue)     AS avg_revenue,
                AVG(lf.walkability_score)    AS avg_walkability_score,
                AVG(lf.restaurant_density)   AS avg_restaurant_density,
                AVG(lf.dist_city_center_km)  AS avg_dist_city_center_km,
                AVG(lf.listings_500m)        AS avg_competition_score
            FROM listing_predictions p
            JOIN listing_features lf ON p.listing_id = lf.listing_id
            GROUP BY p.h3_cell_r8
        """)

        shap_global_arrow = shap_global_df.to_arrow()
        con.register("_shap_global", shap_global_arrow)
        con.execute("CREATE OR REPLACE TABLE shap_global AS SELECT * FROM _shap_global")
        con.unregister("_shap_global")

        hex_shap_arrow = hex_shap_df.to_arrow()
        con.register("_hex_shap", hex_shap_arrow)
        con.execute("CREATE OR REPLACE TABLE hex_shap AS SELECT * FROM _hex_shap")
        con.unregister("_hex_shap")

    n_hexes = preds_df["h3_cell_r8"].n_unique()
    print(f"  Written {len(preds_df)} listing predictions across {n_hexes} H3 cells")
