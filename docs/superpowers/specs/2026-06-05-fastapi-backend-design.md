# FastAPI Backend — Design Spec
**Date:** 2026-06-05  
**Status:** Approved  
**Scope:** Read-only REST API serving pre-computed predictions, H3 hex aggregates, and SHAP values from DuckDB to the React + DeckGL frontend.

---

## 1. Goals

- Serve pre-computed ML predictions and SHAP results to the frontend with <5ms response times.
- Zero cold-start on the API process — all computation happens in the ML pipeline before the server starts.
- Local development only (no Docker, no auth).
- Clean separation: ML pipeline owns computation, API owns serving.

---

## 2. Architecture

### 2.1 Module Layout

```
src/geoai/
  models/
    predict.py        ← NEW: batch predict all listings, aggregate to hex, write SHAP tables
    runner.py         ← EXTEND: add predict step after evaluate
  api/
    __init__.py
    main.py           ← FastAPI app + lifespan (opens DuckDB read-only)
    state.py          ← AppState: holds read-only DuckDB connection
    schemas.py        ← Pydantic response models
    routes/
      __init__.py
      hexagons.py     ← /api/hexagons, /api/hexagons/{hex_id}
      listings.py     ← /api/listings
      shap.py         ← /api/shap/global, /api/shap/{hex_id}
      stats.py        ← /api/stats
```

### 2.2 Pipeline Flow (run before starting API)

```
python -m geoai.models.runner
  1. train_price_model()       → data/models/price_model.pkl
  2. train_occupancy_model()   → data/models/occupancy_model.pkl
  3. run_evaluation()          → prints RMSE / MAE summary
  4. run_predictions()         → writes 4 DuckDB tables (see §3)
```

### 2.3 API Startup

```python
# lifespan hook in main.py
con = duckdb.connect(DB_PATH, read_only=True)
app.state.db = con
```

No model loading. No feature matrix computation. Instant start.

---

## 3. DuckDB Tables (written by `predict.py`)

All tables use `CREATE OR REPLACE TABLE` — safe to re-run after retraining.

### `listing_predictions`
| Column | Type | Notes |
|---|---|---|
| listing_id | VARCHAR | FK to listings.id |
| h3_hex_id | VARCHAR | H3 res-8 hex ID |
| predicted_price | FLOAT | €/night (exp of log prediction) |
| predicted_occupancy | FLOAT | 0–1 |
| predicted_revenue | FLOAT | price × occupancy × 30 |

### `hex_aggregates`
| Column | Type | Notes |
|---|---|---|
| h3_hex_id | VARCHAR | PK |
| avg_price | FLOAT | |
| avg_occupancy | FLOAT | |
| avg_revenue | FLOAT | |
| listing_count | INT | |
| walkability_score | FLOAT | avg across listings in hex |
| transit_score | FLOAT | |
| restaurant_density | FLOAT | |
| competition_score | FLOAT | |

Populated via `GROUP BY h3_hex_id` over `listing_predictions JOIN listing_features`.

### `shap_global`
| Column | Type | Notes |
|---|---|---|
| model | VARCHAR | `'price'` or `'occupancy'` |
| feature | VARCHAR | |
| importance | FLOAT | mean(abs(SHAP)) across all listings |

### `hex_shap`
| Column | Type | Notes |
|---|---|---|
| h3_hex_id | VARCHAR | |
| model | VARCHAR | `'price'` or `'occupancy'` |
| feature | VARCHAR | |
| avg_impact | FLOAT | mean(SHAP value) across listings in hex |
| base_value | FLOAT | explainer.expected_value (same for all rows) |

---

## 4. Endpoints

Base URL: `http://localhost:8000`  
All responses: `application/json`  
All GET endpoints include `Cache-Control: max-age=3600`.  
CORS: allow `http://localhost:5173` (Vite dev server).

### `GET /api/stats`
Market-level KPIs aggregated over all listings.

**Response:**
```json
{
  "avg_price": 87.4,
  "avg_occupancy": 0.71,
  "median_revenue": 1840.0,
  "listing_count": 4218
}
```

### `GET /api/hexagons?mode=price|occupancy|revenue`
All hex cells with the requested value. Drives `H3HexagonLayer` color scale.

**Response:**
```json
[
  { "hex_id": "88395ad6dfffff", "value": 112.5, "listing_count": 23 },
  ...
]
```

### `GET /api/hexagons/{hex_id}`
Full detail for a single hex cell. Drives the right-panel detail view.

**Response:**
```json
{
  "hex_id": "88395ad6dfffff",
  "avg_price": 112.5,
  "avg_occupancy": 0.78,
  "avg_revenue": 2621.0,
  "listing_count": 23,
  "walkability_score": 92.0,
  "transit_score": 85.0,
  "restaurant_density": 0.97,
  "competition_score": 0.61
}
```

### `GET /api/listings?hex_id={hex_id}`
Listing scatter points inside a hex. Drives `ScatterplotLayer`.  
Query: `SELECT id, lat, lng, predicted_price, predicted_occupancy FROM listings JOIN listing_predictions USING(id) WHERE h3_hex_id = ?`

**Response:**
```json
[
  { "id": "12345", "latitude": 41.147, "longitude": -8.611, "predicted_price": 118.0, "predicted_occupancy": 0.80 },
  ...
]
```

### `GET /api/shap/global?model=price|occupancy`
Global feature importance. Drives left-sidebar SHAP bars.

**Response:**
```json
[
  { "feature": "walk_score", "importance": 0.42 },
  { "feature": "restaurant_density", "importance": 0.38 },
  ...
]
```

### `GET /api/shap/{hex_id}?model=price|occupancy`
Per-hex SHAP drivers. Drives right-panel SHAP breakdown.

**Response:**
```json
{
  "hex_id": "88395ad6dfffff",
  "base_value": 74.2,
  "drivers": [
    { "feature": "walk_score", "avg_impact": 18.3 },
    { "feature": "restaurant_density", "avg_impact": 14.1 },
    { "feature": "competition_score", "avg_impact": -9.0 }
  ]
}
```

---

## 5. New Dependencies

Add to `pyproject.toml`:
```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
```

Start command:
```bash
uvicorn geoai.api.main:app --reload --port 8000
```

---

## 6. Error Handling

- `hex_id` not found → `404 Not Found` with `{ "detail": "hex not found" }`
- Invalid `mode` or `model` param → `422 Unprocessable Entity` (FastAPI default)
- DuckDB read error → `500` with detail logged server-side, generic message to client
- Models not trained (pkl missing) → `predict.py` raises `FileNotFoundError` with clear message: run `python -m geoai.models.runner` first

---

## 7. Testing

- Unit tests for `predict.py`: mock DuckDB, assert tables written with correct schema
- Integration tests for each route: spin up `TestClient`, assert response shape + status codes
- No E2E tests (frontend not in scope for this spec)

---

## 8. Out of Scope

- Authentication
- Docker / deployment
- What-if / simulation endpoints
- Multi-city support
- Write endpoints
