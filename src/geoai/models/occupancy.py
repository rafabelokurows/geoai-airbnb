import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from geoai.config import DB_PATH
from geoai.models.features import build_feature_matrix, prepare_X_y_occupancy, split_by_room_type

MODEL_PATH = DB_PATH.parent / "models" / "occupancy_model.pkl"

_MIN_SAMPLES = 50


def train_occupancy_model(db_path: Path = DB_PATH) -> dict:
    df = build_feature_matrix(db_path)

    models: dict = {}
    per_type: dict = {}
    feature_names: list = []

    for group, sub_df in split_by_room_type(df).items():
        if len(sub_df) < _MIN_SAMPLES:
            print(f"  Skipping '{group}' (n={len(sub_df)} < {_MIN_SAMPLES})")
            continue
        X, y, fnames = prepare_X_y_occupancy(sub_df)
        feature_names = list(fnames)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=max(5, len(X_train) // 20),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(50, verbose=False)])
        mae = float(mean_absolute_error(y_test, model.predict(X_test).clip(0, 1)))
        models[group] = model
        per_type[group] = {"mae": mae, "n_train": len(X_train), "n_test": len(X_test)}

    n_train = sum(m["n_train"] for m in per_type.values())
    n_test = sum(m["n_test"] for m in per_type.values())
    weighted_mae = sum(m["mae"] * m["n_test"] for m in per_type.values()) / n_test

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"models": models, "feature_names": feature_names}, f)

    return {
        "mae": weighted_mae,
        "per_type": per_type,
        "n_train": n_train,
        "n_test": n_test,
        "model_path": MODEL_PATH,
    }
