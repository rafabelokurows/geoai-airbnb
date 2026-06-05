import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from geoai.config import DB_PATH
from geoai.models.features import build_feature_matrix, prepare_X_y_price

MODEL_PATH = DB_PATH.parent / "models" / "price_model.pkl"


def train_price_model(db_path: Path = DB_PATH) -> dict:
    df = build_feature_matrix(db_path)
    X, y_log, feature_names = prepare_X_y_price(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    y_pred = np.exp(model.predict(X_test))
    y_true = np.exp(y_test)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "feature_names": list(feature_names)}, f)
    return {
        "rmse": rmse,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model_path": MODEL_PATH,
    }
