# FastAPI Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only FastAPI backend that serves pre-computed H3 hex predictions, SHAP values, and listing scatter data from DuckDB to the React + DeckGL frontend.

**Architecture:** All ML computation (predictions + SHAP) runs in `predict.py` as a pipeline step after model training, writing results to 4 DuckDB tables. The API opens DuckDB in read-only mode and serves from those tables with no cold-start delay.

**Tech Stack:** FastAPI 0.111+, uvicorn, DuckDB (read-only), Polars, NumPy, SHAP, LightGBM, pytest, httpx (via FastAPI TestClient)

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/geoai/models/predict.py` | Batch predict all listings, run SHAP, write 4 DuckDB tables |
| Modify | `src/geoai/models/runner.py` | Call `run_predictions()` after evaluation |
| Modify | `pyproject.toml` | Add fastapi, uvicorn dependencies |
| Create | `src/geoai/api/__init__.py` | Package marker |
| Create | `src/geoai/api/main.py` | FastAPI app, lifespan, CORS, router includes |
| Create | `src/geoai/api/deps.py` | `get_db` dependency (injectable for tests) |
| Create | `src/geoai/api/schemas.py` | Pydantic response models |
| Create | `src/geoai/api/routes/__init__.py` | Package marker |
| Create | `src/geoai/api/routes/stats.py` | `GET /api/stats` |
| Create | `src/geoai/api/routes/hexagons.py` | `GET /api/hexagons`, `GET /api/hexagons/{hex_id}` |
| Create | `src/geoai/api/routes/listings.py` | `GET /api/listings` |
| Create | `src/geoai/api/routes/shap.py` | `GET /api/shap/global`, `GET /api/shap/{hex_id}` |
| Create | `tests/api/__init__.py` | Package marker |
| Create | `tests/api/conftest.py` | Shared test DB fixture + TestClient |
| Create | `tests/models/test_predict.py` | Unit tests for predict.py |
| Create | `tests/api/test_stats.py` | Route tests |
| Create | `tests/api/test_hexagons.py` | Route tests |
| Create | `tests/api/test_listings.py` | Route tests |
| Create | `tests/api/test_shap.py` | Route tests |

---

## Task 1: Add FastAPI + uvicorn dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add dependencies to pyproject.toml**

Open `pyproject.toml` and add to the `dependencies` list:
```toml
[project]
dependencies = [
    "duckdb>=0.10.0",
    "polars>=0.20.0",
    "pyarrow>=15.0.0",
    "osmnx>=1.9.0",
    "geopandas>=0.14.0",
    "shapely>=2.0.0",
    "httpx>=0.27.0",
    "numpy>=1.26.0",
    "h3>=4.0.0",
    "lightgbm>=4.3.0",
    "scikit-learn>=1.4.0",
    "shap>=0.46.0",
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.29.0",
]
```

- [ ] **Step 2: Install**

```bash
.venv\Scripts\pip install -e ".[dev]"
```

- [ ] **Step 3: Verify**

```bash
.venv\Scripts\python -c "import fastapi, uvicorn; print(fastapi.__version__, uvicorn.__version__)"
```

Expected: two version strings printed, no ImportError.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add fastapi and uvicorn dependencies"
```

---

## Task 2: Create `predict.py` — batch predictions + SHAP + write DuckDB tables

**Files:**
- Create: `src/geoai/models/predict.py`
- Create: `tests/models/test_predict.py`

### DuckDB tables written by this module

**`listing_predictions`**
```
listing_id BIGINT, h3_cell_r8 VARCHAR, predicted_price DOUBLE,
predicted_occupancy DOUBLE, predicted_revenue DOUBLE,
latitude DOUBLE, longitude DOUBLE
```

**`hex_aggregates`**
```
h3_cell_r8 VARCHAR, listing_count BIGINT, avg_price DOUBLE,
avg_occupancy DOUBLE, avg_revenue DOUBLE, avg_walkability_score DOUBLE,
avg_restaurant_density DOUBLE, avg_dist_city_center_km DOUBLE,
avg_competition_score DOUBLE
```

**`shap_global`**
```
model VARCHAR, feature VARCHAR, importance DOUBLE
```

**`hex_shap`**
```
h3_cell_r8 VARCHAR, model VARCHAR, feature VARCHAR,
avg_impact DOUBLE, base_value DOUBLE
```

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_predict.py`:

```python
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import numpy as np
import polars as pl
import pytest

from geoai.models.predict import run_predictions, _load_model


def _make_fake_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE listings (
            id BIGINT PRIMARY KEY, latitude DOUBLE, longitude DOUBLE,
            price DOUBLE, accommodates INTEGER, bedrooms DOUBLE, beds DOUBLE,
            room_type VARCHAR, minimum_nights INTEGER, review_scores_rating DOUBLE,
            host_is_superhost BOOLEAN, number_of_reviews INTEGER
        )
    """)
    con.execute("""
        INSERT INTO listings VALUES
        (1, 41.14, -8.61, 100.0, 2, 1.0, 1.0, 'Entire home/apt', 2, 4.8, true, 50),
        (2, 41.15, -8.62, 80.0,  1, 1.0, 1.0, 'Private room',    1, 4.5, false, 30)
    """)
    con.execute("""
        CREATE TABLE listing_features (
            listing_id BIGINT PRIMARY KEY,
            dist_city_center_km DOUBLE, dist_nearest_metro_km DOUBLE,
            dist_nearest_station_km DOUBLE, dist_nearest_supermarket_km DOUBLE,
            dist_airport_km DOUBLE, travel_time_airport_min DOUBLE,
            restaurants_250m INTEGER, restaurants_500m INTEGER, bars_500m INTEGER,
            cafes_500m INTEGER, supermarkets_1km INTEGER, attractions_1km INTEGER,
            museums_2km INTEGER, parks_500m INTEGER, amenity_density_1km INTEGER,
            restaurant_density DOUBLE, listings_500m INTEGER, listings_1km INTEGER,
            avg_price_500m DOUBLE, median_price_neighbourhood DOUBLE,
            walkability_score DOUBLE, dist_livraria_lello_km DOUBLE,
            dist_torre_clerigos_km DOUBLE, dist_ribeira_km DOUBLE,
            dist_ponte_luis_km DOUBLE, dist_mercado_bolhao_km DOUBLE,
            dist_jardins_cristal_km DOUBLE, h3_cell_r8 VARCHAR,
            occupancy_rate_30d DOUBLE, occupancy_rate_90d DOUBLE,
            occupancy_rate_365d DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO listing_features VALUES
        (1, 0.5, 0.3, 0.4, 0.2, 8.0, 20.0, 10, 20, 15, 8, 3, 5, 2, 4, 30, 0.8, 5, 12, 95.0, 90.0, 88.0, 0.3, 0.2, 0.1, 0.4, 0.5, 0.6, '88abc123ffffff', 0.8, 0.75, 0.72),
        (2, 1.2, 0.8, 0.9, 0.6, 9.0, 22.0,  5, 10,  8, 4, 2, 3, 1, 2, 18, 0.5, 3,  8, 80.0, 75.0, 70.0, 0.8, 0.7, 0.6, 0.9, 1.0, 1.1, '88abc456ffffff', 0.6, 0.58, 0.55)
    """)
    con.close()
    return db_path


def _make_fake_model(n_features: int, tmp_path: Path, name: str):
    model = MagicMock()
    model.predict.return_value = np.full(2, 0.5)
    artifact = {"model": model, "feature_names": [f"f{i}" for i in range(n_features)]}
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    return path


class TestLoadModel:
    def test_raises_if_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Run `python -m geoai.models.runner`"):
            _load_model(tmp_path / "missing.pkl")

    def test_loads_artifact(self, tmp_path):
        path = tmp_path / "model.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": "ok"}, f)
        assert _load_model(path) == {"model": "ok"}


class TestRunPredictions:
    def test_creates_all_four_tables(self, tmp_path):
        db_path = _make_fake_db(tmp_path)
        price_path = tmp_path / "models" / "price_model.pkl"
        occ_path = tmp_path / "models" / "occupancy_model.pkl"

        with patch("geoai.models.predict.PRICE_MODEL_PATH", price_path), \
             patch("geoai.models.predict.OCCUPANCY_MODEL_PATH", occ_path), \
             patch("geoai.models.predict.DB_PATH", db_path):

            # Build real-shaped fake models using the actual feature builder
            from geoai.models.features import build_feature_matrix, prepare_X_y_price, prepare_X_y_occupancy
            df = build_feature_matrix(db_path)
            X_p, _, pf = prepare_X_y_price(df)
            X_o, _, of = prepare_X_y_occupancy(df)

            price_model = MagicMock()
            price_model.predict.return_value = np.log(np.full(len(df), 90.0))
            occ_model = MagicMock()
            occ_model.predict.return_value = np.full(len(df), 0.70)

            price_path.parent.mkdir(parents=True, exist_ok=True)
            with open(price_path, "wb") as f:
                pickle.dump({"model": price_model, "feature_names": list(pf)}, f)
            with open(occ_path, "wb") as f:
                pickle.dump({"model": occ_model, "feature_names": list(of)}, f)

            run_predictions(db_path)

        con = duckdb.connect(str(db_path), read_only=True)
        tables = {r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()}
        con.close()

        assert "listing_predictions" in tables
        assert "hex_aggregates" in tables
        assert "shap_global" in tables
        assert "hex_shap" in tables

    def test_listing_predictions_row_count(self, tmp_path):
        db_path = _make_fake_db(tmp_path)
        price_path = tmp_path / "models" / "price_model.pkl"
        occ_path = tmp_path / "models" / "occupancy_model.pkl"

        with patch("geoai.models.predict.PRICE_MODEL_PATH", price_path), \
             patch("geoai.models.predict.OCCUPANCY_MODEL_PATH", occ_path), \
             patch("geoai.models.predict.DB_PATH", db_path):

            from geoai.models.features import build_feature_matrix, prepare_X_y_price, prepare_X_y_occupancy
            df = build_feature_matrix(db_path)
            X_p, _, pf = prepare_X_y_price(df)
            X_o, _, of = prepare_X_y_occupancy(df)

            price_model = MagicMock()
            price_model.predict.return_value = np.log(np.full(len(df), 90.0))
            occ_model = MagicMock()
            occ_model.predict.return_value = np.full(len(df), 0.70)

            price_path.parent.mkdir(parents=True, exist_ok=True)
            with open(price_path, "wb") as f:
                pickle.dump({"model": price_model, "feature_names": list(pf)}, f)
            with open(occ_path, "wb") as f:
                pickle.dump({"model": occ_model, "feature_names": list(of)}, f)

            run_predictions(db_path)

        con = duckdb.connect(str(db_path), read_only=True)
        count = con.execute("SELECT COUNT(*) FROM listing_predictions").fetchone()[0]
        con.close()
        assert count == 2

    def test_shap_global_has_both_models(self, tmp_path):
        db_path = _make_fake_db(tmp_path)
        price_path = tmp_path / "models" / "price_model.pkl"
        occ_path = tmp_path / "models" / "occupancy_model.pkl"

        with patch("geoai.models.predict.PRICE_MODEL_PATH", price_path), \
             patch("geoai.models.predict.OCCUPANCY_MODEL_PATH", occ_path), \
             patch("geoai.models.predict.DB_PATH", db_path):

            from geoai.models.features import build_feature_matrix, prepare_X_y_price, prepare_X_y_occupancy
            df = build_feature_matrix(db_path)
            X_p, _, pf = prepare_X_y_price(df)
            X_o, _, of = prepare_X_y_occupancy(df)

            price_model = MagicMock()
            price_model.predict.return_value = np.log(np.full(len(df), 90.0))
            occ_model = MagicMock()
            occ_model.predict.return_value = np.full(len(df), 0.70)

            price_path.parent.mkdir(parents=True, exist_ok=True)
            with open(price_path, "wb") as f:
                pickle.dump({"model": price_model, "feature_names": list(pf)}, f)
            with open(occ_path, "wb") as f:
                pickle.dump({"model": occ_model, "feature_names": list(of)}, f)

            run_predictions(db_path)

        con = duckdb.connect(str(db_path), read_only=True)
        models = {r[0] for r in con.execute("SELECT DISTINCT model FROM shap_global").fetchall()}
        con.close()
        assert models == {"price", "occupancy"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv\Scripts\python -m pytest tests/models/test_predict.py -v
```

Expected: `ModuleNotFoundError: No module named 'geoai.models.predict'`

- [ ] **Step 3: Implement `src/geoai/models/predict.py`**

```python
import pickle
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import shap as shap_lib

from geoai.config import DB_PATH
from geoai.models.features import build_feature_matrix, prepare_X_y_price, prepare_X_y_occupancy
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

    predicted_price = np.exp(price_artifact["model"].predict(X_price)).astype(np.float64)
    predicted_occupancy = occ_artifact["model"].predict(X_occ).clip(0.0, 1.0).astype(np.float64)
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
    meta = meta.with_columns(
        pl.Series("_order", [id_order.get(i, 999999) for i in meta["listing_id"].to_list()])
    ).sort("_order").drop("_order")

    preds_df = pl.DataFrame({
        "listing_id": listing_ids,
        "h3_cell_r8": meta["h3_cell_r8"].to_list(),
        "predicted_price": predicted_price.tolist(),
        "predicted_occupancy": predicted_occupancy.tolist(),
        "predicted_revenue": predicted_revenue.tolist(),
        "latitude": meta["latitude"].to_list(),
        "longitude": meta["longitude"].to_list(),
    })

    price_explainer = shap_lib.TreeExplainer(
        price_artifact["model"],
        data=shap_lib.sample(X_price, min(100, len(X_price))),
    )
    occ_explainer = shap_lib.TreeExplainer(
        occ_artifact["model"],
        data=shap_lib.sample(X_occ, min(100, len(X_occ))),
    )
    price_shap_vals = price_explainer.shap_values(X_price)
    occ_shap_vals = occ_explainer.shap_values(X_occ)
    price_base = float(price_explainer.expected_value)
    occ_base = float(occ_explainer.expected_value)

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv\Scripts\python -m pytest tests/models/test_predict.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Run full suite to check no regressions**

```bash
.venv\Scripts\python -m pytest --tb=short -q
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/geoai/models/predict.py tests/models/test_predict.py
git commit -m "feat: add batch prediction and SHAP pipeline step"
```

---

## Task 3: Extend `runner.py` to call `run_predictions`

**Files:**
- Modify: `src/geoai/models/runner.py`

- [ ] **Step 1: Update `runner.py`**

Replace the entire file content:

```python
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
```

- [ ] **Step 2: Run existing tests to verify runner still works**

```bash
.venv\Scripts\python -m pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/geoai/models/runner.py
git commit -m "feat: extend runner to call run_predictions after evaluation"
```

---

## Task 4: API scaffold — main.py, deps.py, schemas.py, conftest.py

**Files:**
- Create: `src/geoai/api/__init__.py`
- Create: `src/geoai/api/main.py`
- Create: `src/geoai/api/deps.py`
- Create: `src/geoai/api/schemas.py`
- Create: `src/geoai/api/routes/__init__.py`
- Create: `tests/api/__init__.py`
- Create: `tests/api/conftest.py`

- [ ] **Step 1: Create package markers**

`src/geoai/api/__init__.py` — empty file.
`src/geoai/api/routes/__init__.py` — empty file.
`tests/api/__init__.py` — empty file.

- [ ] **Step 2: Create `src/geoai/api/schemas.py`**

```python
from pydantic import BaseModel


class StatsResponse(BaseModel):
    avg_price: float
    avg_occupancy: float
    median_revenue: float
    listing_count: int


class HexSummary(BaseModel):
    hex_id: str
    value: float
    listing_count: int


class HexDetail(BaseModel):
    hex_id: str
    avg_price: float
    avg_occupancy: float
    avg_revenue: float
    listing_count: int
    avg_walkability_score: float
    avg_restaurant_density: float
    avg_dist_city_center_km: float
    avg_competition_score: float


class ListingPoint(BaseModel):
    id: int
    latitude: float
    longitude: float
    predicted_price: float
    predicted_occupancy: float


class ShapFeature(BaseModel):
    feature: str
    importance: float


class ShapDriver(BaseModel):
    feature: str
    avg_impact: float


class HexShapResponse(BaseModel):
    hex_id: str
    base_value: float
    drivers: list[ShapDriver]
```

- [ ] **Step 3: Create `src/geoai/api/deps.py`**

```python
import duckdb
from fastapi import Request


def get_db(request: Request) -> duckdb.DuckDBPyConnection:
    return request.app.state.db
```

- [ ] **Step 4: Create `src/geoai/api/main.py`**

```python
from contextlib import asynccontextmanager

import duckdb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from geoai.config import DB_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    # read_only=True raises IOException if file doesn't exist — guard for test environments
    if DB_PATH.exists():
        app.state.db = duckdb.connect(str(DB_PATH), read_only=True)
    else:
        app.state.db = None
    yield
    if app.state.db is not None:
        app.state.db.close()


app = FastAPI(title="GeoAI Airbnb API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
```

Routes will be imported and included in later tasks.

- [ ] **Step 5: Create `tests/api/conftest.py`**

```python
import duckdb
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_db(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))

    con.execute("""
        CREATE TABLE listing_predictions (
            listing_id BIGINT, h3_cell_r8 VARCHAR,
            predicted_price DOUBLE, predicted_occupancy DOUBLE,
            predicted_revenue DOUBLE, latitude DOUBLE, longitude DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO listing_predictions VALUES
        (1, '88abc1ffffff', 100.0, 0.75, 2250.0, 41.147, -8.611),
        (2, '88abc1ffffff', 120.0, 0.80, 2880.0, 41.148, -8.612),
        (3, '88abc2ffffff', 80.0,  0.60, 1440.0, 41.150, -8.620)
    """)

    con.execute("""
        CREATE TABLE hex_aggregates (
            h3_cell_r8 VARCHAR PRIMARY KEY,
            listing_count BIGINT,
            avg_price DOUBLE, avg_occupancy DOUBLE, avg_revenue DOUBLE,
            avg_walkability_score DOUBLE, avg_restaurant_density DOUBLE,
            avg_dist_city_center_km DOUBLE, avg_competition_score DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO hex_aggregates VALUES
        ('88abc1ffffff', 2, 110.0, 0.775, 2565.0, 85.0, 0.8, 0.5, 6.0),
        ('88abc2ffffff', 1,  80.0, 0.600, 1440.0, 70.0, 0.5, 1.2, 3.0)
    """)

    con.execute("""
        CREATE TABLE shap_global (
            model VARCHAR, feature VARCHAR, importance DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO shap_global VALUES
        ('price',     'walkability_score',   0.42),
        ('price',     'restaurant_density',  0.38),
        ('occupancy', 'walkability_score',   0.30),
        ('occupancy', 'restaurant_density',  0.25)
    """)

    con.execute("""
        CREATE TABLE hex_shap (
            h3_cell_r8 VARCHAR, model VARCHAR, feature VARCHAR,
            avg_impact DOUBLE, base_value DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO hex_shap VALUES
        ('88abc1ffffff', 'price',     'walkability_score',  15.0, 87.0),
        ('88abc1ffffff', 'price',     'restaurant_density', 10.0, 87.0),
        ('88abc1ffffff', 'occupancy', 'walkability_score',  0.05, 0.71),
        ('88abc1ffffff', 'occupancy', 'restaurant_density', 0.03, 0.71)
    """)

    con.close()
    return db_path


@pytest.fixture
def client(test_db):
    from geoai.api.deps import get_db
    from geoai.api.main import app

    test_con = duckdb.connect(str(test_db), read_only=True)
    app.dependency_overrides[get_db] = lambda: test_con

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
    test_con.close()
```

- [ ] **Step 6: Verify scaffold starts without error**

```bash
.venv\Scripts\python -c "from geoai.api.main import app; print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add src/geoai/api/ tests/api/
git commit -m "feat: scaffold FastAPI app with deps, schemas, and test fixtures"
```

---

## Task 5: `/api/stats` route

**Files:**
- Create: `src/geoai/api/routes/stats.py`
- Create: `tests/api/test_stats.py`
- Modify: `src/geoai/api/main.py`

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_stats.py`:

```python
def test_stats_returns_correct_shape(client):
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"avg_price", "avg_occupancy", "median_revenue", "listing_count"}


def test_stats_listing_count(client):
    resp = client.get("/api/stats")
    assert resp.json()["listing_count"] == 3


def test_stats_avg_price(client):
    resp = client.get("/api/stats")
    avg = resp.json()["avg_price"]
    # test DB: (100 + 120 + 80) / 3 = 100.0
    assert abs(avg - 100.0) < 0.01


def test_stats_cache_header(client):
    resp = client.get("/api/stats")
    assert "max-age=3600" in resp.headers.get("cache-control", "")
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv\Scripts\python -m pytest tests/api/test_stats.py -v
```

Expected: `404 Not Found` or `ImportError` on the route.

- [ ] **Step 3: Create `src/geoai/api/routes/stats.py`**

```python
import duckdb
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from geoai.api.deps import get_db
from geoai.api.schemas import StatsResponse

router = APIRouter()


@router.get("/api/stats", response_model=StatsResponse)
def get_stats(db: duckdb.DuckDBPyConnection = Depends(get_db)):
    row = db.execute("""
        SELECT
            AVG(predicted_price)       AS avg_price,
            AVG(predicted_occupancy)   AS avg_occupancy,
            MEDIAN(predicted_revenue)  AS median_revenue,
            COUNT(*)                   AS listing_count
        FROM listing_predictions
    """).fetchone()
    content = StatsResponse(
        avg_price=row[0],
        avg_occupancy=row[1],
        median_revenue=row[2],
        listing_count=row[3],
    ).model_dump()
    return JSONResponse(content=content, headers={"Cache-Control": "max-age=3600"})
```

- [ ] **Step 4: Register router in `src/geoai/api/main.py`**

Add to main.py after `app.add_middleware(...)`:

```python
from geoai.api.routes.stats import router as stats_router
app.include_router(stats_router)
```

- [ ] **Step 5: Run tests**

```bash
.venv\Scripts\python -m pytest tests/api/test_stats.py -v
```

Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add src/geoai/api/routes/stats.py src/geoai/api/main.py tests/api/test_stats.py
git commit -m "feat: add GET /api/stats endpoint"
```

---

## Task 6: `/api/hexagons` and `/api/hexagons/{hex_id}` routes

**Files:**
- Create: `src/geoai/api/routes/hexagons.py`
- Create: `tests/api/test_hexagons.py`
- Modify: `src/geoai/api/main.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/api/test_hexagons.py`:

```python
import pytest


def test_hexagons_default_mode_price(client):
    resp = client.get("/api/hexagons")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 2
    assert set(items[0].keys()) == {"hex_id", "value", "listing_count"}


def test_hexagons_mode_occupancy(client):
    resp = client.get("/api/hexagons?mode=occupancy")
    assert resp.status_code == 200
    # hex '88abc1ffffff' has avg_occupancy=0.775
    values = {item["hex_id"]: item["value"] for item in resp.json()}
    assert abs(values["88abc1ffffff"] - 0.775) < 0.001


def test_hexagons_mode_revenue(client):
    resp = client.get("/api/hexagons?mode=revenue")
    assert resp.status_code == 200
    values = {item["hex_id"]: item["value"] for item in resp.json()}
    assert abs(values["88abc2ffffff"] - 1440.0) < 0.01


def test_hexagons_invalid_mode(client):
    resp = client.get("/api/hexagons?mode=invalid")
    assert resp.status_code == 422


def test_hexagons_cache_header(client):
    resp = client.get("/api/hexagons")
    assert "max-age=3600" in resp.headers.get("cache-control", "")


def test_hex_detail_valid(client):
    resp = client.get("/api/hexagons/88abc1ffffff")
    assert resp.status_code == 200
    body = resp.json()
    assert body["hex_id"] == "88abc1ffffff"
    assert abs(body["avg_price"] - 110.0) < 0.01
    assert body["listing_count"] == 2
    assert set(body.keys()) == {
        "hex_id", "avg_price", "avg_occupancy", "avg_revenue",
        "listing_count", "avg_walkability_score", "avg_restaurant_density",
        "avg_dist_city_center_km", "avg_competition_score",
    }


def test_hex_detail_not_found(client):
    resp = client.get("/api/hexagons/88notexist")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "hex not found"


def test_hex_detail_cache_header(client):
    resp = client.get("/api/hexagons/88abc1ffffff")
    assert "max-age=3600" in resp.headers.get("cache-control", "")
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv\Scripts\python -m pytest tests/api/test_hexagons.py -v
```

Expected: all fail with 404.

- [ ] **Step 3: Create `src/geoai/api/routes/hexagons.py`**

```python
from typing import Literal

import duckdb
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from geoai.api.deps import get_db
from geoai.api.schemas import HexDetail, HexSummary

router = APIRouter()

_MODE_COLUMN = {
    "price": "avg_price",
    "occupancy": "avg_occupancy",
    "revenue": "avg_revenue",
}


@router.get("/api/hexagons", response_model=list[HexSummary])
def list_hexagons(
    mode: Literal["price", "occupancy", "revenue"] = "price",
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    col = _MODE_COLUMN[mode]
    rows = db.execute(f"""
        SELECT h3_cell_r8 AS hex_id, {col} AS value, listing_count
        FROM hex_aggregates
        ORDER BY hex_id
    """).fetchall()
    content = [HexSummary(hex_id=r[0], value=r[1], listing_count=r[2]).model_dump() for r in rows]
    return JSONResponse(content=content, headers={"Cache-Control": "max-age=3600"})


@router.get("/api/hexagons/{hex_id}", response_model=HexDetail)
def get_hex_detail(
    hex_id: str,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    row = db.execute("""
        SELECT h3_cell_r8, avg_price, avg_occupancy, avg_revenue,
               listing_count, avg_walkability_score, avg_restaurant_density,
               avg_dist_city_center_km, avg_competition_score
        FROM hex_aggregates
        WHERE h3_cell_r8 = ?
    """, [hex_id]).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="hex not found")

    content = HexDetail(
        hex_id=row[0], avg_price=row[1], avg_occupancy=row[2],
        avg_revenue=row[3], listing_count=row[4],
        avg_walkability_score=row[5], avg_restaurant_density=row[6],
        avg_dist_city_center_km=row[7], avg_competition_score=row[8],
    ).model_dump()
    return JSONResponse(content=content, headers={"Cache-Control": "max-age=3600"})
```

- [ ] **Step 4: Register router in `src/geoai/api/main.py`**

Add after the stats router include:

```python
from geoai.api.routes.hexagons import router as hexagons_router
app.include_router(hexagons_router)
```

- [ ] **Step 5: Run tests**

```bash
.venv\Scripts\python -m pytest tests/api/test_hexagons.py -v
```

Expected: `8 passed`

- [ ] **Step 6: Commit**

```bash
git add src/geoai/api/routes/hexagons.py src/geoai/api/main.py tests/api/test_hexagons.py
git commit -m "feat: add GET /api/hexagons and GET /api/hexagons/{hex_id} endpoints"
```

---

## Task 7: `/api/listings` route

**Files:**
- Create: `src/geoai/api/routes/listings.py`
- Create: `tests/api/test_listings.py`
- Modify: `src/geoai/api/main.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/api/test_listings.py`:

```python
def test_listings_for_hex(client):
    resp = client.get("/api/listings?hex_id=88abc1ffffff")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 2
    assert set(items[0].keys()) == {"id", "latitude", "longitude", "predicted_price", "predicted_occupancy"}


def test_listings_hex_with_one_listing(client):
    resp = client.get("/api/listings?hex_id=88abc2ffffff")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_listings_unknown_hex_returns_empty(client):
    resp = client.get("/api/listings?hex_id=88notexist")
    assert resp.status_code == 200
    assert resp.json() == []


def test_listings_missing_hex_id_param(client):
    resp = client.get("/api/listings")
    assert resp.status_code == 422


def test_listings_cache_header(client):
    resp = client.get("/api/listings?hex_id=88abc1ffffff")
    assert "max-age=3600" in resp.headers.get("cache-control", "")
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv\Scripts\python -m pytest tests/api/test_listings.py -v
```

Expected: all fail with 404.

- [ ] **Step 3: Create `src/geoai/api/routes/listings.py`**

```python
import duckdb
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from geoai.api.deps import get_db
from geoai.api.schemas import ListingPoint

router = APIRouter()


@router.get("/api/listings", response_model=list[ListingPoint])
def get_listings(
    hex_id: str,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    rows = db.execute("""
        SELECT listing_id, latitude, longitude, predicted_price, predicted_occupancy
        FROM listing_predictions
        WHERE h3_cell_r8 = ?
        ORDER BY listing_id
    """, [hex_id]).fetchall()
    content = [
        ListingPoint(
            id=r[0], latitude=r[1], longitude=r[2],
            predicted_price=r[3], predicted_occupancy=r[4],
        ).model_dump()
        for r in rows
    ]
    return JSONResponse(content=content, headers={"Cache-Control": "max-age=3600"})
```

- [ ] **Step 4: Register router in `src/geoai/api/main.py`**

Add after the hexagons router include:

```python
from geoai.api.routes.listings import router as listings_router
app.include_router(listings_router)
```

- [ ] **Step 5: Run tests**

```bash
.venv\Scripts\python -m pytest tests/api/test_listings.py -v
```

Expected: `5 passed`

- [ ] **Step 6: Commit**

```bash
git add src/geoai/api/routes/listings.py src/geoai/api/main.py tests/api/test_listings.py
git commit -m "feat: add GET /api/listings endpoint"
```

---

## Task 8: `/api/shap/global` and `/api/shap/{hex_id}` routes

**Files:**
- Create: `src/geoai/api/routes/shap.py`
- Create: `tests/api/test_shap.py`
- Modify: `src/geoai/api/main.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/api/test_shap.py`:

```python
def test_shap_global_price(client):
    resp = client.get("/api/shap/global?model=price")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 2
    assert set(items[0].keys()) == {"feature", "importance"}
    # must be sorted descending by importance
    importances = [i["importance"] for i in items]
    assert importances == sorted(importances, reverse=True)


def test_shap_global_occupancy(client):
    resp = client.get("/api/shap/global?model=occupancy")
    assert resp.status_code == 200
    features = {i["feature"] for i in resp.json()}
    assert "walkability_score" in features


def test_shap_global_invalid_model(client):
    resp = client.get("/api/shap/global?model=banana")
    assert resp.status_code == 422


def test_shap_global_default_is_price(client):
    resp = client.get("/api/shap/global")
    assert resp.status_code == 200


def test_shap_global_cache_header(client):
    resp = client.get("/api/shap/global?model=price")
    assert "max-age=3600" in resp.headers.get("cache-control", "")


def test_shap_hex_valid(client):
    resp = client.get("/api/shap/88abc1ffffff?model=price")
    assert resp.status_code == 200
    body = resp.json()
    assert body["hex_id"] == "88abc1ffffff"
    assert isinstance(body["base_value"], float)
    assert isinstance(body["drivers"], list)
    assert set(body["drivers"][0].keys()) == {"feature", "avg_impact"}
    # sorted by abs(avg_impact) descending
    impacts = [abs(d["avg_impact"]) for d in body["drivers"]]
    assert impacts == sorted(impacts, reverse=True)


def test_shap_hex_not_found(client):
    resp = client.get("/api/shap/88notexist?model=price")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "hex not found"


def test_shap_hex_cache_header(client):
    resp = client.get("/api/shap/88abc1ffffff?model=price")
    assert "max-age=3600" in resp.headers.get("cache-control", "")
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv\Scripts\python -m pytest tests/api/test_shap.py -v
```

Expected: all fail with 404.

- [ ] **Step 3: Create `src/geoai/api/routes/shap.py`**

```python
from typing import Literal

import duckdb
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from geoai.api.deps import get_db
from geoai.api.schemas import HexShapResponse, ShapDriver, ShapFeature

router = APIRouter()


@router.get("/api/shap/global", response_model=list[ShapFeature])
def get_shap_global(
    model: Literal["price", "occupancy"] = "price",
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    rows = db.execute("""
        SELECT feature, importance
        FROM shap_global
        WHERE model = ?
        ORDER BY importance DESC
    """, [model]).fetchall()
    content = [ShapFeature(feature=r[0], importance=r[1]).model_dump() for r in rows]
    return JSONResponse(content=content, headers={"Cache-Control": "max-age=3600"})


@router.get("/api/shap/{hex_id}", response_model=HexShapResponse)
def get_shap_hex(
    hex_id: str,
    model: Literal["price", "occupancy"] = "price",
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    rows = db.execute("""
        SELECT feature, avg_impact, base_value
        FROM hex_shap
        WHERE h3_cell_r8 = ? AND model = ?
        ORDER BY ABS(avg_impact) DESC
    """, [hex_id, model]).fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="hex not found")

    base_value = rows[0][2]
    drivers = [ShapDriver(feature=r[0], avg_impact=r[1]).model_dump() for r in rows]
    content = HexShapResponse(
        hex_id=hex_id,
        base_value=base_value,
        drivers=drivers,
    ).model_dump()
    return JSONResponse(content=content, headers={"Cache-Control": "max-age=3600"})
```

- [ ] **Step 4: Register router in `src/geoai/api/main.py`**

Add after the listings router include:

```python
from geoai.api.routes.shap import router as shap_router
app.include_router(shap_router)
```

- [ ] **Step 5: Run tests**

```bash
.venv\Scripts\python -m pytest tests/api/test_shap.py -v
```

Expected: `8 passed`

- [ ] **Step 6: Run full test suite**

```bash
.venv\Scripts\python -m pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/geoai/api/routes/shap.py src/geoai/api/main.py tests/api/test_shap.py
git commit -m "feat: add GET /api/shap/global and GET /api/shap/{hex_id} endpoints"
```

---

## Final Verification

- [ ] **Smoke test: start the server**

Run the full pipeline first (if not already done):
```bash
.venv\Scripts\python -m geoai.models.runner
```

Then start the API:
```bash
.venv\Scripts\uvicorn geoai.api.main:app --reload --port 8000
```

- [ ] **Verify endpoints respond**

```bash
curl http://localhost:8000/api/stats
curl "http://localhost:8000/api/hexagons?mode=price"
curl http://localhost:8000/docs
```

Expected: JSON responses and interactive docs at `/docs`.

- [ ] **Final commit tag**

```bash
git tag v0.5.0-api
```
